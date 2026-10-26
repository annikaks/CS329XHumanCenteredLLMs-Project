"""TF-IDF (word + char n-grams) + logistic regression for --task rachel|partner.

C is chosen by episode-grouped CV on train only; the test split is touched once.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.evaluation import partner_metrics, rachel_metrics


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 lowercase=False, max_features=200_000)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rachel", "partner"], required=True)
    ap.add_argument("--data-dir", default="data/classifiers")
    ap.add_argument("--out-dir", default="results/classifiers")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--C", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    args = ap.parse_args()

    train = load(os.path.join(args.data_dir, f"{args.task}_train.jsonl"))
    test = load(os.path.join(args.data_dir, f"{args.task}_test.jsonl"))
    X_tr, y_tr = [r["text"] for r in train], [r["label"] for r in train]
    groups = [r["episode"] for r in train]

    scoring = "roc_auc" if args.task == "rachel" else "f1_macro"
    search = GridSearchCV(build_pipeline(), {"clf__C": args.C}, cv=GroupKFold(n_splits=5),
                          scoring=scoring, n_jobs=-1, refit=True)
    search.fit(X_tr, y_tr, groups=groups)
    model = search.best_estimator_
    proba = model.predict_proba([r["text"] for r in test])
    classes = list(model.classes_)

    if args.task == "rachel":
        metrics = rachel_metrics(test, proba[:, classes.index("Rachel")])
    else:
        metrics = partner_metrics(test, proba, classes)
    metrics = {"task": args.task, "model": "tfidf_word1-2_char2-5_logreg", "best_C": search.best_params_["clf__C"],
               f"cv_{scoring}": search.best_score_, "n_train": len(train), "n_test": len(test),
               "classes": classes, **metrics}

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.task}_tfidf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, os.path.join(args.model_dir, f"{args.task}_tfidf.joblib"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
