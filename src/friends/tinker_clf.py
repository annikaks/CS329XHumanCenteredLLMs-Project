"""Classification as next-token prediction on Tinker (LoRA on a base LLM).

Each class gets a letter. The prompt lists the options and ends with "Answer:"; the model is trained to
emit " A", " B", ... with loss only on that token. Scoring is one prefill per example: the sampler returns
log-probs for every option letter at the answer position (`target_prompt_logprobs`), which are
renormalized over the option set to get class probabilities.
"""
import json
import os
import string

import numpy as np

PARTNERS = ["Chandler", "Joey", "Monica", "Other", "Phoebe", "Ross"]

TASKS = {
    "rachel": {
        "labels": ["Rachel", "Other"],
        "display": {"Rachel": "Rachel", "Other": "someone else"},
        "instruction": "Who said this line from Friends? Character names are masked as [NAME].",
        "field": "Line",
    },
    "partner": {
        "labels": PARTNERS,
        "display": {"Other": "someone outside the main six"},
        "instruction": ("This is something Rachel says in a two-person conversation from Friends. Who is she "
                        "talking to? Character names are masked as [NAME]."),
        "field": "Rachel",
    },
}

REGISTRY = os.path.join(os.path.dirname(__file__), "..", "..", "results", "classifiers", "tinker_checkpoints.json")


def letters(task):
    return list(string.ascii_uppercase[: len(TASKS[task]["labels"])])


def build_prompt(task, text):
    spec = TASKS[task]
    opts = "\n".join(f"{l}) {spec['display'].get(lab, lab)}" for l, lab in zip(letters(task), spec["labels"]))
    return f"{spec['instruction']}\n\nOptions:\n{opts}\n\n{spec['field']}:\n{text}\n\nAnswer:"


def label_token_ids(tokenizer, task):
    ids = [tokenizer.encode(" " + l, add_special_tokens=False) for l in letters(task)]
    assert all(len(x) == 1 for x in ids), f"option letters must be single tokens, got {ids}"
    flat = [x[0] for x in ids]
    assert len(set(flat)) == len(flat), "option letter tokens collide"
    return flat


def encode_prompt(tokenizer, task, text, max_text_tokens=1024):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(text_ids) > max_text_tokens:
        text = tokenizer.decode(text_ids[:max_text_tokens])
    return tokenizer.encode(build_prompt(task, text), add_special_tokens=True)


def make_datum(types, prompt_ids, label_id, weight=1.0):
    tokens = prompt_ids + [label_id]
    weights = np.zeros(len(tokens) - 1, dtype=np.float32)
    weights[-1] = weight
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": np.array(tokens[1:], dtype=np.int64), "weights": weights},
    )


def score_batch(types, sampling_client, prompts, label_ids, chunk=64):
    """prompts: list of token-id lists (no answer). Returns [N, K] probabilities over the option set."""
    import torch

    K = len(label_ids)
    out = np.zeros((len(prompts), K), dtype=np.float64)
    for start in range(0, len(prompts), chunk):
        futures = []
        for p in prompts[start:start + chunk]:
            full = p + [label_ids[0]]  # placeholder answer token so the answer position is inside the prompt
            ids = torch.full((len(full) - 1, K), -1, dtype=torch.int64)
            ids[len(p) - 1] = torch.tensor(label_ids)
            futures.append(sampling_client.sample(
                prompt=types.ModelInput.from_ints(full),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
                target_prompt_logprobs=types.TensorData.from_torch_sparse(ids, pad_value=-1),
            ))
        for j, (p, fut) in enumerate(zip(prompts[start:start + chunk], futures)):
            lp = fut.result().target_prompt_logprobs.to_torch()[len(p) - 1].double().numpy()
            lp = lp - lp.max()
            out[start + j] = np.exp(lp) / np.exp(lp).sum()
    return out


class TinkerClassifier:
    """sklearn-style wrapper (classes_, predict_proba) around a saved Tinker sampler checkpoint."""

    def __init__(self, task, sampler_path=None, registry=REGISTRY):
        import tinker
        from tinker import types

        if sampler_path is None:
            with open(registry) as f:
                sampler_path = json.load(f)[task]["sampler_path"]
        self.task, self.types = task, types
        self.classes_ = np.array(TASKS[task]["labels"])
        service = tinker.ServiceClient()
        self.sampler = service.create_sampling_client(model_path=sampler_path)
        self.tokenizer = self.sampler.get_tokenizer()
        self.label_ids = label_token_ids(self.tokenizer, task)

    def predict_proba(self, texts):
        prompts = [encode_prompt(self.tokenizer, self.task, t) for t in texts]
        return score_batch(self.types, self.sampler, prompts, self.label_ids)
