"""Score generated Person A responses with the trained classifiers.

Input: a results JSON like results/baseline/mistral_mixed_fewshot_baseline.json
(fields: context, gold_response, generated_response, actual_partner).

speaker: P(Rachel | response), gold vs generated.
pair:    context + response as an S1/S2 transcript -> P(true pair), and accuracy when
         the decision is restricted to {Rachel-Ross, Monica-Rachel}. The context is
         shared, so the gold-vs-generated gap is what the single response contributes.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.masking import mask_names

PARTNER_TO_PAIR = {"Person B": "Rachel-Ross", "Person C": "Monica-Rachel"}


def as_transcript(context, response):
    turns = context + [{"speaker": "Person A", "text": response}]
    order = list(dict.fromkeys(t["speaker"] for t in turns))
    alias = {sp: f"S{i + 1}" for i, sp in enumerate(order)}
    texts = mask_names([t["text"] for t in turns])
    return "\n".join(f"{alias[t['speaker']]}: {x}" for t, x in zip(turns, texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/baseline/mistral_mixed_fewshot_baseline.json")
    ap.add_argument("--speaker-model", default="models/speaker_tfidf.joblib")
    ap.add_argument("--pair-model", default="models/pair_tfidf.joblib")
    ap.add_argument("--out", default="results/classifiers/baseline_generation_scores.json")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)
    spk = joblib.load(args.speaker_model)
    pair = joblib.load(args.pair_model)
    rachel = list(spk.classes_).index("Rachel")
    pclasses = list(pair.classes_)
    two = [pclasses.index("Rachel-Ross"), pclasses.index("Monica-Rachel")]

    summary, per_example = {}, []
    for field in ["gold_response", "generated_response"]:
        resp = mask_names([r[field] for r in rows])
        ps = spk.predict_proba(resp)
        trans = [as_transcript(r["context"], r[field]) for r in rows]
        pp = pair.predict_proba(trans)
        truth = [pclasses.index(PARTNER_TO_PAIR[r["actual_partner"]]) for r in rows]
        p_true = pp[np.arange(len(rows)), truth]
        two_way_pred = [two[i] for i in pp[:, two].argmax(1)]
        summary[field] = {
            "speaker_mean_P_rachel": float(ps[:, rachel].mean()),
            "speaker_top1_rachel_rate": float((ps.argmax(1) == rachel).mean()),
            "pair_mean_P_true_pair": float(p_true.mean()),
            "pair_top1_acc_15way": float((pp.argmax(1) == np.array(truth)).mean()),
            "pair_acc_restricted_AB_vs_AC": float((np.array(two_way_pred) == np.array(truth)).mean()),
        }
        for i, r in enumerate(rows):
            if len(per_example) <= i:
                per_example.append({"conversation_id": r["conversation_id"], "actual_partner": r["actual_partner"]})
            per_example[i][field] = {
                "P_rachel": float(ps[i, rachel]),
                "speaker_top1": spk.classes_[ps[i].argmax()],
                "P_true_pair": float(p_true[i]),
                "pair_top1": pclasses[pp[i].argmax()],
            }

    out = {"n": len(rows), "summary": summary, "examples": per_example}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
