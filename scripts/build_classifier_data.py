"""Build datasets for the two classifiers.

  speaker: one utterance by one of the six mains -> who said it (6-way)
  pair:    a pair-only segment (both speakers relabeled S1/S2) -> which pair (15-way)

Scenes in data/processed/friends_split.json["test_episode_ids"] are held out for
generation testing and never enter either classifier. The remaining scenes are
split train/test by *episode*, shared by both tasks, so no episode is on both sides.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.corpus import ALL_PAIRS, MAIN_SIX, episode_of, extract_pair_segments, load_conversations
from friends.masking import mask_names

SHORT = {
    "Rachel Green": "Rachel", "Ross Geller": "Ross", "Monica Geller": "Monica",
    "Chandler Bing": "Chandler", "Joey Tribbiani": "Joey", "Phoebe Buffay": "Phoebe",
}


def pair_label(pair):
    return "-".join(SHORT[p] for p in sorted(pair))


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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    convos = load_conversations(args.source, args.raw_dir)

    with open(args.heldout) as f:
        heldout = set(json.load(f)["test_episode_ids"])
    known = {cid for cid, _ in convos}
    missing = heldout - known
    assert not missing, f"held-out ids not in corpus: {sorted(missing)[:5]}"

    pool = [(cid, turns) for cid, turns in convos if cid not in heldout]

    episodes = sorted({episode_of(cid) for cid, _ in pool})
    rng = random.Random(args.seed)
    test_eps = set(rng.sample(episodes, round(len(episodes) * args.test_frac)))
    split_of = lambda cid: "test" if episode_of(cid) in test_eps else "train"

    # Task 1: speaker
    spk_rows = [
        {"conversation_id": cid, "episode": episode_of(cid), "label": SHORT[sp], "text_raw": text}
        for cid, turns in pool
        for sp, text in turns
        if sp in MAIN_SIX and text and text.strip()
    ]
    masked = mask_names([r["text_raw"] for r in spk_rows], use_ner=not args.no_ner)
    for r, m in zip(spk_rows, masked):
        r["text"] = m
        r["n_words"] = len(m.split())

    # Task 2: pair
    pair_rows = []
    for pair in ALL_PAIRS:
        for seg in extract_pair_segments(pool, pair):
            order = list(dict.fromkeys(t["speaker"] for t in seg["turns"]))
            alias = {sp: f"S{i + 1}" for i, sp in enumerate(order)}
            pair_rows.append({
                "conversation_id": seg["conversation_id"],
                "episode": episode_of(seg["conversation_id"]),
                "label": pair_label(pair),
                "turns_raw": seg["turns"],
                "speakers": [alias[t["speaker"]] for t in seg["turns"]],
            })
    flat = [t["text"] for r in pair_rows for t in r["turns_raw"]]
    flat_masked = iter(mask_names(flat, use_ner=not args.no_ner))
    for r in pair_rows:
        lines = [f"{s}: {next(flat_masked)}" for s in r["speakers"]]
        r["text"] = "\n".join(lines)
        r["n_turns"] = len(lines)
        del r["speakers"]

    stats = {"heldout_scenes_excluded": len(heldout), "n_episodes": len(episodes), "n_test_episodes": len(test_eps)}
    for name, rows in [("speaker", spk_rows), ("pair", pair_rows)]:
        for split in ["train", "test"]:
            part = [r for r in rows if split_of(r["conversation_id"]) == split]
            write_jsonl(part, os.path.join(args.out_dir, f"{name}_{split}.jsonl"))
            stats[f"{name}_{split}"] = {"n": len(part), "labels": dict(sorted(Counter(r["label"] for r in part).items()))}

    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"seed": args.seed, "test_episodes": sorted(test_eps), "heldout_scenes": sorted(heldout)}, f, indent=2)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
