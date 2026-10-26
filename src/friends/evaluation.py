"""Metrics shared by every classifier backend (TF-IDF, RoBERTa, Tinker) for the two tasks.

All CIs are cluster bootstraps over scenes (conversation_id): lines from one scene are not independent.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

LENGTH_BUCKETS = [(1, 3), (4, 7), (8, 15), (16, 10_000)]


def _cluster_bootstrap(clusters, stat, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([index[c] for c in rng.choice(uniq, len(uniq))])
        try:
            v = stat(idx)
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def rachel_metrics(rows, p_rachel, k_values=(1, 3, 5, 10), seed=0):
    """rows need label, conversation_id, speaker, n_words. p_rachel: P(Rachel) per row."""
    y = np.array([r["label"] == "Rachel" for r in rows])
    p = np.asarray(p_rachel, dtype=float)
    scenes = [r["conversation_id"] for r in rows]
    nw = np.array([r["n_words"] for r in rows])
    pred = p >= 0.5
    out = {
        "auc": _auc(y, p),
        "auc_ci95": _cluster_bootstrap(scenes, lambda i: _auc(y[i], p[i]), seed=seed),
        "accuracy@0.5": float((pred == y).mean()),
        "balanced_accuracy@0.5": float((pred[y].mean() + (~pred[~y]).mean()) / 2),
        "base_rate_rachel": float(y.mean()),
        "auc_by_length": {
            f"{lo}-{hi if hi < 10_000 else 'inf'}w": {"auc": _auc(y[m], p[m]), "n": int(m.sum())}
            for lo, hi in LENGTH_BUCKETS for m in [(nw >= lo) & (nw <= hi)]
        },
    }
    # Average log-odds over k lines by the same speaker in the same scene (judging a multi-line generation).
    rng = np.random.default_rng(seed)
    logit = np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - p, 1e-9, 1))
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["conversation_id"], r["speaker"]), []).append(i)
    agg = {}
    for k in k_values:
        ys, ss = [], []
        for (_, spk), idx in groups.items():
            idx = rng.permutation(idx)
            for s in range(0, len(idx) - k + 1, k):
                ys.append(spk == "Rachel")
                ss.append(logit[idx[s:s + k]].mean())
        agg[f"k={k}"] = {"auc": _auc(ys, ss), "n": len(ys)}
    out["auc_aggregating_k_lines"] = agg
    return out


def partner_metrics(rows, proba, classes, seed=0):
    """rows need label, conversation_id, segment_id. proba: [N, K] over `classes` (one row per Rachel line)."""
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)
    y = np.array([classes.index(r["label"]) for r in rows])
    scenes = np.array([r["conversation_id"] for r in rows])

    def block(y_, P_, scenes_):
        pred = P_.argmax(1)
        maj = np.bincount(y_, minlength=len(classes)).argmax()
        res = {
            "n": int(len(y_)),
            "accuracy": float(accuracy_score(y_, pred)),
            "accuracy_ci95": _cluster_bootstrap(scenes_, lambda i: float((P_[i].argmax(1) == y_[i]).mean()), seed=seed),
            "macro_f1": float(f1_score(y_, pred, average="macro", labels=list(range(len(classes))))),
            "majority_baseline": float((y_ == maj).mean()),
            "chance_uniform": 1 / len(classes),
            "per_partner": {},
        }
        for k, c in enumerate(classes):
            yk = y_ == k
            if yk.sum() == 0:
                continue
            res["per_partner"][c] = {
                "n": int(yk.sum()),
                "ovr_auc": _auc(yk, P_[:, k]),
                "ovr_auc_ci95": _cluster_bootstrap(scenes_, lambda i: _auc(yk[i], P_[i, k]), seed=seed),
                # ceiling for a generator: mean P(c) on lines to c minus mean P(c) on lines to others
                "gap": float(P_[yk, k].mean() - P_[~yk, k].mean()),
                "gap_ci95": _cluster_bootstrap(
                    scenes_, lambda i: (P_[i][yk[i], k].mean() - P_[i][~yk[i], k].mean()) if yk[i].any() and (~yk[i]).any() else None,
                    seed=seed),
            }
        return res

    out = {"line_level": block(y, proba, scenes)}

    # Segment level: sum log-probs over all of Rachel's lines in a segment.
    segs = {}
    for i, r in enumerate(rows):
        segs.setdefault(r["segment_id"], []).append(i)
    logp = np.log(np.clip(proba, 1e-12, 1))
    seg_ids = list(segs)
    S = np.array([logp[segs[s]].sum(0) for s in seg_ids])
    S = np.exp(S - S.max(1, keepdims=True))
    S /= S.sum(1, keepdims=True)
    y_seg = np.array([y[segs[s][0]] for s in seg_ids])
    scenes_seg = np.array([scenes[segs[s][0]] for s in seg_ids])
    out["segment_level"] = block(y_seg, S, scenes_seg)
    return out
