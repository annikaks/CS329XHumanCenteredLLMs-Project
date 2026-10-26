"""LoRA-fine-tune a base LLM on Tinker as the `rachel` (is this Rachel?) or `partner` (who is Rachel talking to?)
classifier, with persistent checkpoints.

  export TINKER_API_KEY=...
  python scripts/train_tinker.py --list-models
  python scripts/train_tinker.py --task partner
  python scripts/train_tinker.py --task rachel

Before training, the untrained base model is scored on val with the same prompt ("epoch -1"). If that is far
above chance, the model is recognizing memorized Friends lines, and test scores will overstate how well it
judges new (generated) text.

Training runs on Tinker's GPUs; nothing runs locally except tokenization and bookkeeping.
Checkpoints: a resumable save_state every --save-every steps (expires after --state-ttl-days), a sampler
checkpoint per epoch, an HF-format export of the best epoch's LoRA adapter downloaded to models/tinker/<run>/,
and a final state. Every path is appended to results/classifiers/tinker_runs/<run>.jsonl as it is written,
so a crashed run can be resumed with the --resume line logged there.

Same episode-grouped val carve-out as train_transformer.py. Each epoch saves a sampler checkpoint and scores
val; the best epoch's tinker:// path is evaluated once on test and written to
results/classifiers/tinker_checkpoints.json so later code can load it by task name
(friends.tinker_clf.TinkerClassifier("partner")).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from friends.evaluation import partner_metrics, rachel_metrics
from friends.tinker_clf import REGISTRY, TASKS, encode_prompt, label_token_ids, make_datum, score_batch

# rachel: ~35k short lines -> enough data for rank 16, two passes.
# partner: ~7.5k lines (relaxed train segments) -> still small: lower rank and LR, best epoch kept by val.
DEFAULTS = {
    "rachel": {"epochs": 2, "batch_size": 128, "lr": 2e-4, "rank": 16, "val_n": 1500},
    "partner": {"epochs": 3, "batch_size": 32, "lr": 1e-4, "rank": 8, "val_n": 100_000},
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_service_client():
    import tinker
    return tinker.ServiceClient()


def download_external(service, tinker_path, dest):
    """Download an exported (HF-format) LoRA checkpoint to `dest` via signed URLs."""
    import urllib.request

    urls = service.create_rest_client().get_external_weights_urls(tinker_path).result().urls
    for rel, url in urls.items():
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        urllib.request.urlretrieve(url, out)
    return sorted(urls)


def log_checkpoint(log_path, **entry):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")


def evaluate(task, types, sampler, prompts, y, labels, label_ids):
    """Selection score: rachel -> AUC of P(Rachel); partner -> mean one-vs-rest AUC over partners."""
    proba = score_batch(types, sampler, prompts, label_ids)
    if task == "rachel":
        score = roc_auc_score(y == labels.index("Rachel"), proba[:, labels.index("Rachel")])
    else:
        score = float(np.mean([roc_auc_score(y == k, proba[:, k]) for k in range(len(labels)) if 0 < (y == k).sum() < len(y)]))
    return proba, {"score": float(score), "acc": float((proba.argmax(1) == y).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rachel", "partner"])
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-n", type=int, help="cap val examples scored per epoch")
    ap.add_argument("--skip-base-eval", action="store_true", help="skip scoring the untrained model on val")
    ap.add_argument("--max-train", type=int, default=None, help="subsample train (cost/debug)")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--data-dir", default="data/classifiers")
    ap.add_argument("--out-dir", default="results/classifiers")
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=50, help="steps between resumable save_state checkpoints")
    ap.add_argument("--resume", default=None, help="tinker://.../weights/... state path to resume from")
    ap.add_argument("--resume-step", type=int, default=0, help="global step the --resume checkpoint was saved at")
    ap.add_argument("--run-name", default=None, help="reuse the original run name when resuming")
    ap.add_argument("--no-export", action="store_true", help="skip exporting/downloading the best LoRA adapter")
    ap.add_argument("--model-dir", default="models/tinker")
    ap.add_argument("--state-ttl-days", type=float, default=30, help="expiry for intermediate states (finals never expire)")
    ap.add_argument("--dry-run", action="store_true", help="count tokens with the HF tokenizer; no Tinker calls")
    args = ap.parse_args(argv)

    from tinker import types

    if args.dry_run:
        return dry_run(args)
    service = make_service_client()
    if args.list_models:
        for m in service.get_server_capabilities().supported_models:
            print(m)
        return
    assert args.task, "--task is required"
    for k, v in DEFAULTS[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    rng = random.Random(args.seed)
    labels = TASKS[args.task]["labels"]
    label2id = {l: i for i, l in enumerate(labels)}

    train = load(os.path.join(args.data_dir, f"{args.task}_train.jsonl"))
    test = load(os.path.join(args.data_dir, f"{args.task}_test.jsonl"))
    eps = sorted({r["episode"] for r in train})
    val_eps = set(rng.sample(eps, max(1, round(len(eps) * args.val_frac))))
    val = [r for r in train if r["episode"] in val_eps]
    train = [r for r in train if r["episode"] not in val_eps]
    if args.max_train and len(train) > args.max_train:
        train = rng.sample(train, args.max_train)
    if len(val) > args.val_n:
        val = rng.sample(val, args.val_n)

    run_name = args.run_name or f"{args.task}-{args.base_model.split('/')[-1]}-r{args.rank}-{time.strftime('%Y%m%d-%H%M%S')}"
    ckpt_log = os.path.join(args.out_dir, "tinker_runs", f"{run_name}.jsonl")
    if args.resume:
        tc = service.create_training_client_from_state_with_optimizer(
            args.resume, user_metadata={"project": "cs329x-friends", "task": args.task, "run": run_name})
    else:
        tc = service.create_lora_training_client(
            base_model=args.base_model, rank=args.rank, seed=args.seed,
            user_metadata={"project": "cs329x-friends", "task": args.task, "run": run_name},
        )
    print(f"base={args.base_model} rank={args.rank} lr={args.lr} epochs={args.epochs} batch={args.batch_size}")
    tok = tc.get_tokenizer()
    label_ids = label_token_ids(tok, args.task)

    def encode(rows):
        return [encode_prompt(tok, args.task, r["text"]) for r in rows], np.array([label2id[r["label"]] for r in rows])

    tr_prompts, tr_y = encode(train)
    va_prompts, va_y = encode(val)
    te_prompts, te_y = encode(test)

    counts = np.bincount(tr_y, minlength=len(labels))
    class_w = np.ones(len(labels)) if args.no_class_weights else counts.sum() / (len(labels) * np.maximum(counts, 1))

    n_steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    print(f"{run_name}: train={len(train)} val={len(val)} test={len(test)} steps={n_steps} "
          f"max_prompt_tokens={max(map(len, tr_prompts))}\ncheckpoint log: {ckpt_log}")

    history, best = [], None
    if not args.skip_base_eval and not args.resume:
        base_sampler = service.create_sampling_client(base_model=args.base_model)
        _, base_m = evaluate(args.task, types, base_sampler, va_prompts, va_y, labels, label_ids)
        history.append({"epoch": -1, "step": 0, "sampler_path": None, **{f"val_{k}": v for k, v in base_m.items()}})
        log_checkpoint(ckpt_log, kind="base_eval", **history[-1])
        print("untrained base model on val:", history[-1], flush=True)
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            if step < args.resume_step:  # same seed -> same order; skip what the resumed state already saw
                step += 1
                continue
            idx = order[start:start + args.batch_size]
            batch = [make_datum(types, tr_prompts[i], label_ids[tr_y[i]], float(class_w[tr_y[i]])) for i in idx]
            lr = args.lr * (1 - step / n_steps)
            fb = tc.forward_backward(batch, "cross_entropy")
            opt = tc.optim_step(adam_params=types.AdamParams(learning_rate=lr))
            out = fb.result()
            opt.result()
            if step % 10 == 0:
                try:
                    nll = f"{-np.mean([o['logprobs'].to_numpy()[-1] for o in out.loss_fn_outputs]):.4f}"
                except Exception:
                    nll = out.metrics
                print(f"epoch {epoch} step {step}/{n_steps} lr {lr:.2e} answer_nll {nll}", flush=True)
            step += 1
            if args.save_every and step % args.save_every == 0 and step < n_steps:
                path = tc.save_state(name=f"{run_name}-step{step}",
                                     ttl_seconds=int(args.state_ttl_days * 86400) or None).result().path
                log_checkpoint(ckpt_log, kind="state", step=step, path=path,
                               resume=f"--resume {path} --resume-step {step} --run-name {run_name}")

        ckpt = tc.save_weights_for_sampler(name=f"{run_name}-ep{epoch}").result().path
        sampler = service.create_sampling_client(model_path=ckpt)
        _, val_m = evaluate(args.task, types, sampler, va_prompts, va_y, labels, label_ids)
        history.append({"epoch": epoch, "step": step, "sampler_path": ckpt,
                        **{f"val_{k}": v for k, v in val_m.items()}})
        log_checkpoint(ckpt_log, kind="sampler", **history[-1])
        print(history[-1], flush=True)
        if best is None or val_m["score"] > best["val_score"]:
            best = history[-1]
            if not args.no_export:
                best["external_path"] = tc.save_weights_external(name=f"{run_name}-ep{epoch}-hf").result().path
                log_checkpoint(ckpt_log, kind="external", epoch=epoch, path=best["external_path"])

    state_path = tc.save_state(name=f"{run_name}-final").result().path
    log_checkpoint(ckpt_log, kind="state", step=step, path=state_path, final=True)
    local_dir = None
    if not args.no_export:
        local_dir = os.path.join(args.model_dir, run_name)
        files = download_external(service, best["external_path"], local_dir)
        print(f"downloaded best adapter (epoch {best['epoch']}) to {local_dir}: {files}")
    sampler = service.create_sampling_client(model_path=best["sampler_path"])
    proba, test_m = evaluate(args.task, types, sampler, te_prompts, te_y, labels, label_ids)
    if args.task == "rachel":
        test_detail = rachel_metrics(test, proba[:, labels.index("Rachel")])
    else:
        test_detail = partner_metrics(test, proba, labels)

    tag = f"{args.task}_tinker-{args.base_model.split('/')[-1]}"
    metrics = {
        "task": args.task, "model": f"tinker-lora:{args.base_model}", "rank": args.rank, "lr": args.lr,
        "epochs": args.epochs, "batch_size": args.batch_size, "class_weights": not args.no_class_weights,
        "n_train": len(train), "n_val": len(val), "n_test": len(test), "history": history,
        "best_sampler_path": best["sampler_path"], "final_state_path": state_path,
        "classes": labels, "test_score": test_m["score"], "test_accuracy": test_m["acc"],
        "base_model_val": next((h for h in history if h["epoch"] == -1), None), **test_detail,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    registry = {}
    if os.path.exists(args.registry):
        with open(args.registry) as f:
            registry = json.load(f)
    registry[args.task] = {
        "sampler_path": best["sampler_path"], "state_path": state_path, "base_model": args.base_model,
        "rank": args.rank, "run": run_name, "best_epoch": best["epoch"],
        "external_path": best.get("external_path"), "local_adapter_dir": local_dir,
        "val_score": best["val_score"], "test_score": test_m["score"], "test_accuracy": test_m["acc"],
    }
    with open(args.registry, "w") as f:
        json.dump(registry, f, indent=2)
    print(json.dumps(metrics, indent=2))


def dry_run(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)
    for task in [args.task] if args.task else ["partner", "rachel"]:
        epochs = args.epochs or DEFAULTS[task]["epochs"]
        train = load(os.path.join(args.data_dir, f"{task}_train.jsonl"))
        test = load(os.path.join(args.data_dir, f"{task}_test.jsonl"))
        lens = np.array([len(encode_prompt(tok, task, r["text"])) for r in train])
        test_tok = sum(len(encode_prompt(tok, task, r["text"])) for r in test)
        n_val = min(args.val_n or DEFAULTS[task]["val_n"], round(len(train) * args.val_frac))
        n_val_evals = epochs + 1  # + untrained base model
        train_tok = lens.sum() * (1 - args.val_frac) * epochs
        sample_tok = lens.mean() * n_val * n_val_evals + test_tok
        print(f"{task}: train ~{train_tok / 1e6:.2f}M tokens ({epochs} epochs), sampling ~{sample_tok / 1e6:.2f}M "
              f"(val each epoch + test once). Multiply by {args.base_model}'s prices on the Tinker pricing page.")


if __name__ == "__main__":
    main()
