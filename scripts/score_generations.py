"""Score generated Rachel (Person A) responses with the two classifiers.

Input: a results JSON like results/baseline/mistral_mixed_fewshot_baseline.json
(fields: gold_response, generated_response, actual_partner in {"Person B", "Person C"}).

rachel:  P(Rachel | response), gold vs generated.
partner: P(partner | response) from Rachel's line alone (no context: the context is real dialogue and would
         give the partner away). Reported per true partner, plus the gap
         mean P(Ross) in Ross scenes - mean P(Ross) in Monica scenes, for gold vs generated.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.masking import mask_names

PARTNER_OF = {"Person B": "Ross", "Person C": "Monica"}


class HFClassifier:
    """Wraps a fine-tuned checkpoint dir with the sklearn predict_proba/classes_ interface."""

    def __init__(self, path, max_len=64, batch_size=64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        cfg = self.model.config.id2label
        self.classes_ = np.array([cfg[i] for i in range(len(cfg))])
        self.max_len, self.batch_size = max_len, batch_size

    def predict_proba(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(list(texts[i:i + self.batch_size]), truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                out.append(self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)


def load_model(path, task):
    """.joblib (TF-IDF), an HF checkpoint dir, a tinker:// sampler path, or "tinker" (use the registry)."""
    if path.endswith(".joblib"):
        return joblib.load(path)
    if path == "tinker" or path.startswith("tinker://"):
        from friends.tinker_clf import TinkerClassifier
        return TinkerClassifier(task, sampler_path=None if path == "tinker" else path)
    return HFClassifier(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/baseline/mistral_mixed_fewshot_baseline.json")
    ap.add_argument("--rachel-model", default="models/rachel_tfidf.joblib", help=".joblib, HF dir, tinker://..., or 'tinker'")
    ap.add_argument("--partner-model", default="models/partner_tfidf.joblib", help=".joblib, HF dir, tinker://..., or 'tinker'")
    ap.add_argument("--out", default="results/classifiers/baseline_generation_scores.json")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)
    rachel = load_model(args.rachel_model, "rachel")
    partner = load_model(args.partner_model, "partner")
    r_idx = list(rachel.classes_).index("Rachel")
    pclasses = list(partner.classes_)
    truth = np.array([PARTNER_OF[r["actual_partner"]] for r in rows])

    summary, examples = {}, [{"conversation_id": r["conversation_id"], "partner": t} for r, t in zip(rows, truth)]
    for field in ["gold_response", "generated_response"]:
        texts = mask_names([r[field] for r in rows])
        p_r = rachel.predict_proba(texts)[:, r_idx]
        P = partner.predict_proba(texts)
        p_true = P[np.arange(len(rows)), [pclasses.index(t) for t in truth]]
        ross, mon = pclasses.index("Ross"), pclasses.index("Monica")
        summary[field] = {
            "mean_P_rachel": float(p_r.mean()),
            "mean_P_true_partner": float(p_true.mean()),
            "ross_gap": float(P[truth == "Ross", ross].mean() - P[truth == "Monica", ross].mean()),
            "monica_gap": float(P[truth == "Monica", mon].mean() - P[truth == "Ross", mon].mean()),
            "ross_vs_monica_acc": float(((P[:, ross] > P[:, mon]) == (truth == "Ross")).mean()),
        }
        for e, a, b, c in zip(examples, p_r, p_true, P):
            e[field] = {"P_rachel": float(a), "P_true_partner": float(b),
                        "partner_probs": {k: round(float(v), 4) for k, v in zip(pclasses, c)}}

    out = {"n": len(rows), "summary": summary, "examples": examples}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
