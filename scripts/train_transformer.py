"""Fine-tune a HF encoder (e.g. RoBERTa) for --task rachel|partner on any GPU.

  python scripts/train_transformer.py --task rachel --model roberta-base
  python scripts/train_transformer.py --task partner --model roberta-base --epochs 6

A small validation set (episode-grouped, carved from train) picks the best epoch;
the test split is evaluated once at the end.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TextDS(Dataset):
    def __init__(self, rows, label2id):
        self.texts = [r["text"] for r in rows]
        self.labels = [label2id[r["label"]] for r in rows]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tok, max_len):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
        logits.append(out.logits.float().cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rachel", "partner"], required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--data-dir", default="data/classifiers")
    ap.add_argument("--out-dir", default="results/classifiers")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=None, help="default 64")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap rows per split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    max_len = args.max_len or 64

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load(os.path.join(args.data_dir, f"{args.task}_train.jsonl"))
    test = load(os.path.join(args.data_dir, f"{args.task}_test.jsonl"))
    eps = sorted({r["episode"] for r in train})
    val_eps = set(random.Random(args.seed).sample(eps, max(1, round(len(eps) * args.val_frac))))
    val = [r for r in train if r["episode"] in val_eps]
    train = [r for r in train if r["episode"] not in val_eps]
    if args.limit:
        train, val, test = (random.sample(x, min(args.limit, len(x))) for x in (train, val, test))

    labels = sorted({r["label"] for r in train})
    label2id = {l: i for i, l in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens": ["[NAME]"]})
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    collate = make_collate(tok, max_len)
    dl_train = DataLoader(TextDS(train, label2id), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_val = DataLoader(TextDS(val, label2id), batch_size=args.batch_size * 2, collate_fn=collate)
    dl_test = DataLoader(TextDS(test, label2id), batch_size=args.batch_size * 2, collate_fn=collate)

    counts = np.bincount([label2id[r["label"]] for r in train], minlength=len(labels))
    class_w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_train) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    tag = f"{args.task}_{args.model.split('/')[-1]}"
    ckpt = os.path.join(args.model_dir, tag)
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dl_train):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(dl_train)} loss {loss.item():.4f}")
        lv, yv = predict(model, dl_val, device)
        f1 = f1_score(yv, lv.argmax(1), average="macro")
        history.append({"epoch": epoch, "val_acc": accuracy_score(yv, lv.argmax(1)), "val_macro_f1": f1})
        print(history[-1])
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)

    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    lt, yt = predict(model, dl_test, device)
    pred = lt.argmax(1)
    metrics = {
        "task": args.task, "model": args.model, "max_len": max_len, "epochs": args.epochs, "lr": args.lr,
        "n_train": len(train), "n_val": len(val), "n_test": len(test), "history": history,
        "test_accuracy": accuracy_score(yt, pred),
        "test_macro_f1": f1_score(yt, pred, average="macro"),
        "chance_uniform": 1 / len(labels),
        "per_class": classification_report(yt, pred, labels=list(range(len(labels))), target_names=labels,
                                           output_dict=True, zero_division=0),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
