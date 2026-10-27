"""Per-conversation report from score_generations.py output: Rachel-voice probability and the partner each
response was classified as, for the real (gold) and generated Rachel line. Standard library only.

  python scripts/final_report.py \
      --results results/baseline/mistral_mixed_fewshot_baseline.json \
      --scores results/classifiers/baseline_generation_scores_final.json \
      --out results/classifiers/final_report
Writes <out>.csv (one row per conversation) and <out>.md (summary + table).
"""
import argparse
import csv
import json
import random


def classify(partner_probs):
    top = max(partner_probs, key=partner_probs.get)
    rm = "Ross" if partner_probs["Ross"] > partner_probs["Monica"] else "Monica"
    return top, rm


def paired_ci(pairs, n_boot=5000, seed=0):
    rng = random.Random(seed)
    diffs = sorted(sum(a - b for a, b in (rng.choice(pairs) for _ in pairs)) / len(pairs) for _ in range(n_boot))
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/baseline/mistral_mixed_fewshot_baseline.json")
    ap.add_argument("--scores", default="results/classifiers/baseline_generation_scores_final.json")
    ap.add_argument("--out", default="results/classifiers/final_report")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        texts = json.load(f)
    with open(args.scores, encoding="utf-8") as f:
        scores = json.load(f)["examples"]
    assert len(texts) == len(scores)

    rows = []
    for t, s in zip(texts, scores):
        assert t["conversation_id"] == s["conversation_id"]
        row = {"conversation_id": s["conversation_id"], "true_partner": s["partner"]}
        for kind, field in [("gold", "gold_response"), ("gen", "generated_response")]:
            top, rm = classify(s[field]["partner_probs"])
            row.update({
                f"{kind}_text": t[field].replace("\n", " "),
                f"{kind}_P_rachel": round(s[field]["P_rachel"], 3),
                f"{kind}_partner_top": top,
                f"{kind}_ross_vs_monica": rm,
                f"{kind}_ross_vs_monica_correct": rm == s["partner"],
                f"{kind}_P_true_partner": round(s[field]["P_true_partner"], 3),
            })
        rows.append(row)

    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean = lambda k, rs=rows: sum(float(r[k]) for r in rs) / len(rs)
    pr = [(r["gold_P_rachel"], r["gen_P_rachel"]) for r in rows]
    rm = [(int(r["gold_ross_vs_monica_correct"]), int(r["gen_ross_vs_monica_correct"])) for r in rows]
    lo1, hi1 = paired_ci(pr)
    lo2, hi2 = paired_ci(rm)

    def confusion(kind):
        out = {}
        for p in ["Ross", "Monica"]:
            rs = [r for r in rows if r["true_partner"] == p]
            out[p] = {c: sum(r[f"{kind}_ross_vs_monica"] == c for r in rs) for c in ["Ross", "Monica"]}
        return out

    lines = [
        f"# Final scoring: {n} conversations",
        "",
        "Judges: Rachel voice = Qwen3-8B LoRA (Tinker); partner = TF-IDF + logistic regression (6 partners).",
        "",
        "| | Real Rachel line | Generated line | Difference (95% CI) |",
        "|---|---|---|---|",
        f"| Mean P(Rachel's voice) | {mean('gold_P_rachel'):.2f} | {mean('gen_P_rachel'):.2f} | "
        f"{mean('gold_P_rachel') - mean('gen_P_rachel'):+.2f} ({lo1:+.2f} to {hi1:+.2f}) |",
        f"| Lines judged Rachel (P > 0.5) | {sum(r['gold_P_rachel'] > 0.5 for r in rows)}/{n} | "
        f"{sum(r['gen_P_rachel'] > 0.5 for r in rows)}/{n} | |",
        f"| Ross vs Monica classified correctly | {mean('gold_ross_vs_monica_correct'):.2f} | "
        f"{mean('gen_ross_vs_monica_correct'):.2f} | {mean('gold_ross_vs_monica_correct') - mean('gen_ross_vs_monica_correct'):+.2f} "
        f"({lo2:+.2f} to {hi2:+.2f}) |",
        f"| Mean P(true partner), 6-way | {mean('gold_P_true_partner'):.2f} | {mean('gen_P_true_partner'):.2f} | |",
        "",
        "Ross vs Monica classification (rows = true partner):",
        "",
        "| | Real: →Ross | Real: →Monica | Generated: →Ross | Generated: →Monica |",
        "|---|---|---|---|---|",
    ]
    cg, cn = confusion("gold"), confusion("gen")
    for p in ["Ross", "Monica"]:
        lines.append(f"| {p} | {cg[p]['Ross']} | {cg[p]['Monica']} | {cn[p]['Ross']} | {cn[p]['Monica']} |")
    lines += ["", "## Per conversation", "",
              "| id | partner | real P(R) | real → | gen P(R) | gen → | generated line |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        txt = r["gen_text"][:90].replace("|", "/") + ("…" if len(r["gen_text"]) > 90 else "")
        lines.append(f"| {r['conversation_id']} | {r['true_partner']} | {r['gold_P_rachel']:.2f} | {r['gold_ross_vs_monica']} | "
                     f"{r['gen_P_rachel']:.2f} | {r['gen_ross_vs_monica']} | {txt} |")
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
