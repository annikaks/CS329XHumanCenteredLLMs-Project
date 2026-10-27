"""Offline checks for the Tinker classifier path (no network): fake clients, real tinker.types.

  TOKENIZER_DIR=/path/to/any/hf/tokenizer python -m pytest tests/test_tinker_offline.py -q
"""
import json
import os
import sys
from concurrent.futures import Future

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

tinker = pytest.importorskip("tinker")
from tinker import types  # noqa: E402

from friends import tinker_clf  # noqa: E402


def _tokenizer():
    from transformers import AutoTokenizer
    path = os.environ.get("TOKENIZER_DIR")
    if not path:
        pytest.skip("set TOKENIZER_DIR")
    return AutoTokenizer.from_pretrained(path)


def _done(x):
    f = Future()
    f.set_result(x)
    return f


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSampler:
    """Puts the most mass on option (len(prompt) % K), so tests can predict the argmax."""

    def __init__(self, tok):
        self.tok, self.calls = tok, 0

    def get_tokenizer(self):
        return self.tok

    def sample(self, prompt, num_samples, sampling_params, include_prompt_logprobs, target_prompt_logprobs):
        self.calls += 1
        n = prompt.length if hasattr(prompt, "length") else len(prompt.to_ints())
        ids = target_prompt_logprobs.to_torch(pad_value=-1)
        assert ids.shape[0] == n - 1
        rows = (ids != -1).any(1).nonzero().flatten().tolist()
        assert rows == [n - 2], rows  # only the answer position is scored
        K = ids.shape[1]
        lp = torch.full((n - 1, K), -1e9, dtype=torch.float32)
        lp[n - 2] = torch.full((K,), -5.0)
        lp[n - 2, (n - 1) % K] = -0.1
        return _done(Obj(target_prompt_logprobs=types.TensorData.from_torch(lp)))


class FakeTraining:
    def __init__(self, tok):
        self.tok, self.steps, self.batches = tok, 0, []

    def get_tokenizer(self):
        return self.tok

    def forward_backward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.batches.append(data)
        outs = [{"logprobs": types.TensorData.from_numpy(np.full(d.model_input.length, -1.0, dtype=np.float32))}
                for d in data]
        return _done(Obj(loss_fn_outputs=outs, metrics={}))

    def optim_step(self, adam_params):
        self.steps += 1
        return _done(Obj())

    def save_weights_for_sampler(self, name):
        return _done(Obj(path=f"tinker://fake/sampler_weights/{name}"))

    def save_state(self, name, ttl_seconds=None):
        self.states = getattr(self, "states", []) + [name]
        return _done(Obj(path=f"tinker://fake/weights/{name}"))

    def save_weights_external(self, name):
        self.exports = getattr(self, "exports", []) + [name]
        return _done(Obj(path=f"tinker://fake/external_weights/{name}"))


class FakeRest:
    def __init__(self, root):
        self.root = root

    def get_external_weights_urls(self, tinker_path):
        name = tinker_path.rsplit("/", 1)[-1]
        src = self.root / name
        src.mkdir(parents=True, exist_ok=True)
        (src / "adapter_config.json").write_text(json.dumps({"r": 32}))
        (src / "adapter_model.safetensors").write_bytes(b"fake")
        return _done(Obj(urls={p.name: p.as_uri() for p in src.iterdir()}, expires=None))


class FakeService:
    def __init__(self, tok, root=None):
        self.tok, self.training, self.root, self.resumed = tok, FakeTraining(tok), root, None

    def create_lora_training_client(self, **kw):
        return self.training

    def create_training_client_from_state_with_optimizer(self, path, user_metadata=None):
        self.resumed = path
        return self.training

    def create_rest_client(self):
        return FakeRest(self.root)

    def create_sampling_client(self, model_path=None, base_model=None):
        assert (model_path or "").startswith("tinker://") or base_model
        self.sampling_calls = getattr(self, "sampling_calls", []) + [model_path or f"base:{base_model}"]
        return FakeSampler(self.tok)


def _rows(task, n=60):
    labels = tinker_clf.TASKS[task]["labels"]
    return [{"text": f"line {i} " + "word " * (i % 7), "label": labels[i % len(labels)],
             "episode": f"s01_e{i % 12:02d}", "conversation_id": f"s01_e{i % 12:02d}_c{i % 3:02d}_u001",
             "segment_id": f"s01_e{i % 12:02d}_c{i % 3:02d}_u001#{i % 2}", "speaker": "Rachel" if i % 2 else "Ross",
             "n_words": 2 + i % 7} for i in range(n)]


def test_label_tokens_and_datum():
    tok = _tokenizer()
    for task in ["rachel", "partner"]:
        ids = tinker_clf.label_token_ids(tok, task)
        assert len(ids) == len(tinker_clf.TASKS[task]["labels"])
        p = tinker_clf.encode_prompt(tok, task, "Oh. My. God. [NAME]!")
        d = tinker_clf.make_datum(types, p, ids[1], weight=2.0)
        w = d.loss_fn_inputs["weights"].to_numpy()
        t = d.loss_fn_inputs["target_tokens"].to_numpy()
        assert d.model_input.length == len(p) == len(w) == len(t)
        assert w[-1] == 2.0 and w[:-1].sum() == 0 and t[-1] == ids[1]
        assert "Answer:" in tok.decode(p[-3:])


def test_dry_run(capsys):
    import train_tinker
    train_tinker.main(["--dry-run", "--base-model", os.environ.get("TOKENIZER_DIR") or pytest.skip("set TOKENIZER_DIR"),
                       "--data-dir", os.path.join(ROOT, "data", "classifiers")])
    out = capsys.readouterr().out
    assert "partner: train ~" in out and "rachel: train ~" in out


def test_score_batch_reads_answer_position():
    tok = _tokenizer()
    ids = tinker_clf.label_token_ids(tok, "partner")
    prompts = [tinker_clf.encode_prompt(tok, "partner", "word " * n) for n in range(20)]
    proba = tinker_clf.score_batch(types, FakeSampler(tok), prompts, ids, chunk=7)
    assert np.allclose(proba.sum(1), 1)
    assert (proba.argmax(1) == np.array([len(p) % len(ids) for p in prompts])).all()


@pytest.mark.parametrize("task,unit", [("partner", "line"), ("rachel", "line"), ("partner", "segment")])
def test_train_loop_end_to_end(task, unit, tmp_path, monkeypatch):
    tok = _tokenizer()
    import train_tinker

    data = tmp_path / "data"
    data.mkdir()
    for split in ["train", "test"]:
        (data / f"{task}_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in _rows(task)))
    fake = FakeService(tok, root=tmp_path / "remote")
    monkeypatch.setattr(train_tinker, "make_service_client", lambda: fake)
    reg = tmp_path / "reg.json"
    common = ["--task", task, "--epochs", "2", "--batch-size", "16" if unit == "line" else "4", "--data-dir", str(data),
              "--out-dir", str(tmp_path / "out"), "--registry", str(reg), "--model-dir", str(tmp_path / "models"),
              "--val-frac", "0.25", "--unit", unit]
    train_tinker.main(common + ["--save-every", "3"])
    entry = json.loads(reg.read_text())[task]
    assert fake.sampling_calls[0].startswith("base:")  # untrained model scored first
    assert entry["sampler_path"].startswith("tinker://fake/sampler_weights/")
    assert entry["state_path"].startswith("tinker://fake/weights/")
    assert entry["rank"] == train_tinker.DEFAULTS[task]["rank"]
    n_train_steps = fake.training.steps
    assert n_train_steps == len(fake.training.batches) and n_train_steps >= 4
    assert any(n.endswith("-step3") for n in fake.training.states)
    assert entry["external_path"].startswith("tinker://fake/external_weights/")
    assert sorted(os.listdir(entry["local_adapter_dir"])) == ["adapter_config.json", "adapter_model.safetensors"]
    log = [json.loads(l) for l in open(tmp_path / "out" / "tinker_runs" / f"{entry['run']}.jsonl")]
    assert {"base_eval", "state", "sampler", "external"} <= {e["kind"] for e in log}
    tag = task + ("-seg" if unit == "segment" else "")
    m = json.loads((tmp_path / "out" / f"{tag}_tinker-Qwen3-8B_metrics.json").read_text())
    if unit == "segment":
        assert m["n_test"] == len({r["segment_id"] for r in _rows(task)}) and entry["unit"] == "segment"
    assert [h["epoch"] for h in m["history"]] == [-1, 0, 1]
    assert m["base_model_val"]["epoch"] == -1
    assert ("auc" in m) if task == "rachel" else ("segment_level" in m and "per_partner" in m["line_level"])

    resume_from = next(e for e in log if e["kind"] == "state" and e["step"] == 3)
    fake2 = FakeService(tok, root=tmp_path / "remote2")
    monkeypatch.setattr(train_tinker, "make_service_client", lambda: fake2)
    train_tinker.main(common + ["--resume", resume_from["path"], "--resume-step", "3", "--run-name", entry["run"]])
    assert fake2.resumed == resume_from["path"]
    assert fake2.training.steps == n_train_steps - 3
    assert not any(c.startswith("base:") for c in fake2.sampling_calls)

    monkeypatch.setattr(tinker, "ServiceClient", lambda: fake)
    clf = tinker_clf.TinkerClassifier(task, registry=str(reg))
    p = clf.predict_proba(["Oh. My. God.", "No way!"])
    assert p.shape == (2, len(tinker_clf.TASKS[task]["labels"]))
    assert list(clf.classes_) == tinker_clf.TASKS[task]["labels"]
