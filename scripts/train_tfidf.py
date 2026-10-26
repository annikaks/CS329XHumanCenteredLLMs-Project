"""TF-IDF (word + char n-grams) + logistic regression for --task speaker|pair.

C is chosen by episode-grouped CV on train only; the test split is touched once.
"""
import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 lowercase=False, max_features=200_000)),
    ])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    return Pipeline([("feats", feats), ("clf", clf)])


def aggregate_eval(rows, proba, classes, k_values=(1, 3, 5, 10), seed=0):
    """Accuracy when averaging log-probs over k utterances from the same speaker+scene.
    Mirrors judging a generated multi-turn transcript rather than a single line."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["conversation_id"], r["label"]), []).append(i)
    logp = np.log(proba + 1e-12)
    out = {}
    for k in k_values:
        correct = total = 0
        for (_, label), idx in groups.items():
            if len(idx) < k:
                continue
            idx = rng.permutation(idx)
            for start in range(0, len(idx) - k + 1, k):
                chunk = idx[start:start + k]
                pred = classes[logp[chunk].sum(0).argmax()]
                correct += pred == label
                total += 1
        out[f"k={k}"] = {"acc": correct / total if total else None, "n": total}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["speaker", "pair"], required=True)
    ap.add_argument("--data-dir", default="data/classifiers")
    ap.add_argument("--out-dir", default="results/classifiers")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--min-words", type=int, default=0, help="speaker task: drop shorter utterances from train AND test")
    ap.add_argument("--C", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    args = ap.parse_args()

    train = load(os.path.join(args.data_dir, f"{args.task}_train.jsonl"))
    test = load(os.path.join(args.data_dir, f"{args.task}_test.jsonl"))
    if args.task == "speaker" and args.min_words:
        train = [r for r in train if r["n_words"] >= args.min_words]
        test = [r for r in test if r["n_words"] >= args.min_words]

    X_tr, y_tr = [r["text"] for r in train], [r["label"] for r in train]
    X_te, y_te = [r["text"] for r in test], [r["label"] for r in test]
    groups = [r["episode"] for r in train]

    search = GridSearchCV(
        build_pipeline(), {"clf__C": args.C}, cv=GroupKFold(n_splits=5),
        scoring="f1_macro", n_jobs=-1, refit=True,
    )
    search.fit(X_tr, y_tr, groups=groups)
    model = search.best_estimator_
    classes = model.classes_

    proba = model.predict_proba(X_te)
    pred = classes[proba.argmax(1)]
    majority = max(set(y_tr), key=y_tr.count)

    tag = args.task + (f"_min{args.min_words}w" if args.min_words else "")
    metrics = {
        "task": args.task,
        "model": "tfidf_word1-2_char2-5_logreg",
        "best_C": search.best_params_["clf__C"],
        "cv_macro_f1": search.best_score_,
        "n_train": len(train), "n_test": len(test),
        "test_accuracy": accuracy_score(y_te, pred),
        "test_macro_f1": f1_score(y_te, pred, average="macro"),
        "chance_uniform": 1 / len(classes),
        "majority_class_acc": float(np.mean(np.array(y_te) == majority)),
        "per_class": classification_report(y_te, pred, output_dict=True, zero_division=0),
    }
    if args.task == "speaker":
        buckets = [(1, 3), (4, 7), (8, 15), (16, 10_000)]
        n_words = np.array([r["n_words"] for r in test])
        metrics["acc_by_length"] = {
            f"{lo}-{hi if hi < 10_000 else 'inf'}w": {
                "acc": float(np.mean(pred[(n_words >= lo) & (n_words <= hi)] == np.array(y_te)[(n_words >= lo) & (n_words <= hi)])),
                "n": int(((n_words >= lo) & (n_words <= hi)).sum()),
            }
            for lo, hi in buckets
        }
        metrics["acc_aggregating_k_utterances"] = aggregate_eval(test, proba, classes)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{tag}_tfidf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, os.path.join(args.model_dir, f"{tag}_tfidf.joblib"))

    fig, ax = plt.subplots(figsize=(7, 6) if args.task == "speaker" else (12, 11))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=classes, normalize="true",
                                            values_format=".2f", ax=ax, xticks_rotation=60, colorbar=False)
    ax.set_title(f"{tag}: acc={metrics['test_accuracy']:.3f}, macro-F1={metrics['test_macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{tag}_tfidf_confusion.png"), dpi=130)

    summary = {k: v for k, v in metrics.items() if k != "per_class"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
