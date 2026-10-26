"""Build datasets for the two Rachel-centred classifiers.

  rachel:  one line by one of the six mains -> "Rachel" or "Other"          (is this Rachel?)
  partner: one Rachel line from a Rachel-only-with-X segment -> X           (who is she talking to?)
           X in {Ross, Monica, Chandler, Joey, Phoebe, Other}; "Other" = any non-main character.
           Only Rachel's lines are kept; the partner's lines would give the answer away.
           Train uses a relaxed segment rule (>=2 turns, >=1 each: ~1.7x more lines, better test scores);
           test keeps the notebook's strict rule (>=4 turns, >=2 each) so its labels are as clean as possible.

Scenes in data/processed/friends_split.json["test_episode_ids"] are held out for generation testing and
never enter either classifier. The rest is split train/test by *episode*, shared by both tasks.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.corpus import MAIN_SIX, episode_of, extract_pair_segments, load_conversations
from friends.masking import mask_names

RACHEL = "Rachel Green"
SHORT = {
    "Rachel Green": "Rachel", "Ross Geller": "Ross", "Monica Geller": "Monica",
    "Chandler Bing": "Chandler", "Joey Tribbiani": "Joey", "Phoebe Buffay": "Phoebe",
}
NON_SPEAKERS = {"NONE", "#ALL#", "All", "Everyone"}


def partner_label(speaker):
    return SHORT.get(speaker, "Other")


def rachel_segments(convos, min_turns=4, min_each=2):
    """Contiguous runs where only Rachel and one partner speak, for every partner in the scene."""
    segs = []
    for cid, turns in convos:
        partners = {sp for sp, _ in turns if sp != RACHEL and sp not in NON_SPEAKERS}
        if not any(sp == RACHEL for sp, _ in turns):
            continue
        for p in sorted(partners):
            segs.extend(extract_pair_segments([(cid, turns)], {RACHEL, p}, min_turns, min_each))
    return segs


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["emory", "convokit"], default="emory")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--heldout", default="data/processed/friends_split.json")
    ap.add_argument("--out-dir", default="data/classifiers")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--train-min-turns", type=int, default=2)
    ap.add_argument("--train-min-each", type=int, default=1)
    ap.add_argument("--test-min-turns", type=int, default=4)
    ap.add_argument("--test-min-each", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    convos = load_conversations(args.source, args.raw_dir)

    with open(args.heldout) as f:
        heldout = set(json.load(f)["test_episode_ids"])
    missing = heldout - {cid for cid, _ in convos}
    assert not missing, f"held-out ids not in corpus: {sorted(missing)[:5]}"
    pool = [(cid, turns) for cid, turns in convos if cid not in heldout]

    episodes = sorted({episode_of(cid) for cid, _ in pool})
    rng = random.Random(args.seed)
    test_eps = set(rng.sample(episodes, round(len(episodes) * args.test_frac)))
    split_of = lambda cid: "test" if episode_of(cid) in test_eps else "train"

    # Task 1: is this line Rachel's?
    rachel_rows = [
        {"conversation_id": cid, "episode": episode_of(cid), "speaker": SHORT[sp],
         "label": "Rachel" if sp == RACHEL else "Other", "text_raw": text}
        for cid, turns in pool
        for sp, text in turns
        if sp in MAIN_SIX and text and text.strip()
    ]

    # Task 2: who is Rachel talking to? (Rachel's lines only)
    partner_rows = []
    train_pool = [(c, t) for c, t in pool if split_of(c) == "train"]
    test_pool = [(c, t) for c, t in pool if split_of(c) == "test"]
    segments = (rachel_segments(train_pool, args.train_min_turns, args.train_min_each)
                + rachel_segments(test_pool, args.test_min_turns, args.test_min_each))
    for k, seg in enumerate(segments):
        partner = next(t["speaker"] for t in seg["turns"] if t["speaker"] != RACHEL)
        seg_id = f"{seg['conversation_id']}#{k}"
        for i, t in enumerate(seg["turns"]):
            if t["speaker"] == RACHEL and t["text"].strip():
                partner_rows.append({
                    "conversation_id": seg["conversation_id"], "episode": episode_of(seg["conversation_id"]),
                    "segment_id": seg_id, "turn_index": i, "label": partner_label(partner),
                    "partner_raw": partner, "text_raw": t["text"],
                })

    for rows in (rachel_rows, partner_rows):
        for r, m in zip(rows, mask_names([r["text_raw"] for r in rows], use_ner=not args.no_ner)):
            r["text"] = m
            r["n_words"] = len(m.split())

    stats = {"partner_segment_rule": {"train": [args.train_min_turns, args.train_min_each],
                                      "test": [args.test_min_turns, args.test_min_each]},
             "heldout_scenes_excluded": len(heldout), "n_episodes": len(episodes), "n_test_episodes": len(test_eps)}
    for name, rows in [("rachel", rachel_rows), ("partner", partner_rows)]:
        for split in ["train", "test"]:
            part = [r for r in rows if split_of(r["conversation_id"]) == split]
            write_jsonl(part, os.path.join(args.out_dir, f"{name}_{split}.jsonl"))
            s = {"n": len(part), "labels": dict(sorted(Counter(r["label"] for r in part).items()))}
            if name == "partner":
                s["segments"] = dict(sorted(Counter(r["label"] for r in {r["segment_id"]: r for r in part}.values()).items()))
            stats[f"{name}_{split}"] = s

    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"seed": args.seed, "test_episodes": sorted(test_eps), "heldout_scenes": sorted(heldout)}, f, indent=2)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
