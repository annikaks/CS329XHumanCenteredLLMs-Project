# CS 329X: Partner-conditioned dialogue on *Friends*

Does a model's imitation of Rachel change depending on who she's talking to? This repo has the data pipeline,
the few-shot Mistral baseline, and two classifiers used as automatic judges of generated Rachel lines.

## Layout

```
notebooks/
  329XData.ipynb        ConvoKit -> Rachel–Ross / Rachel–Monica pair segments, anonymization, scene-level split
  BaselineRun.ipynb     Mistral-7B-Instruct few-shot generation (pilot: 20 A–B + 20 A–C test examples)
data/processed/
  friends_AB_anonymized.json, friends_AC_anonymized.json   481 / 295 segments
  friends_split.json    80/20 split of scene ids; test_episode_ids = HELD OUT for generation testing
data/classifiers/       built by scripts/build_classifier_data.py
results/baseline/       Mistral outputs
results/classifiers/    metrics JSON, Tinker checkpoint registry and run logs
src/friends/            corpus loading, segment extraction, name masking, metrics, Tinker classifier
scripts/                build data, train (TF-IDF / Tinker / RoBERTa), score generations
tests/                  offline tests for the Tinker path
```

## The two classifiers

| | 1. `rachel`: is this line Rachel's? | 2. `partner`: who is Rachel talking to? |
|---|---|---|
| Input | one line by any of the six mains | one Rachel line from a scene where only she and X speak |
| Labels | Rachel / Other | Ross, Monica, Chandler, Joey, Phoebe, Other (any non-main character) |
| Train / test lines | 38,923 / 9,116 (17% Rachel) | 8,319 / 1,056 (4,208 / 287 segments) |
| Used for | does a generated line sound like Rachel? | does generated Rachel shift by partner the way real Rachel does? |

`partner` sees **only Rachel's lines**. The partner's own lines identify the partner, and in generation
experiments those are real context, so including them would score the context rather than the generation.

**Data decisions**
- **Held-out scenes are excluded entirely.** The 111 scenes in `friends_split.json["test_episode_ids"]` are
  dropped first. Neither classifier sees the scenes we generate from.
- **Train/test is split by episode** (`sXX_eYY`, 47 of 236 episodes in test), shared by both tasks. Scenes in an
  episode share plot, so a scene-level split would reward topic recognition.
- **Names are masked to `[NAME]`.** This covers the six mains and their nicknames, the notebook's `Person X`,
  `[Other Person]`, and spaCy PERSON entities, with a stoplist so style markers like *Y'know* survive.
- Segments are contiguous turns where only Rachel and one partner speak. **Test** uses the notebook's rule
  (≥4 turns, ≥2 each; Rachel–Ross / Rachel–Monica counts match the notebook). **Train** relaxes it to ≥2 turns,
  ≥1 each, for 1.7× more lines. On the same strict test set that raised partner line accuracy from 0.224 to 0.251
  and Monica's AUC from 0.52 to 0.57, so the extra data outweighs its noisier labels.
- CIs are cluster bootstraps over scenes, because lines from one scene aren't independent.

**Source data.** `329XData.ipynb` uses ConvoKit's `friends-corpus`. The scripts default to the
[emorynlp/character-mining](https://github.com/emorynlp/character-mining) JSON that ConvoKit's corpus was built
from (`speaker = speakers[0]`), which reproduces ConvoKit's counts and the 481/295 segments exactly.
`--source convokit` reads ConvoKit directly.

## TF-IDF baseline results (word + char n-grams, logistic regression, C by episode-grouped CV)

**`rachel`**: AUC 0.66 (95% CI 0.64–0.68) on a single line; by length: 1–3 words 0.64, 16+ words 0.73.
Averaging log-odds over 3 / 5 / 10 lines from the same scene: AUC 0.77 / 0.79 / 0.85.

**`partner`** (6-way, chance 0.17, majority 0.24 line / 0.20 scene). Accuracy 0.25 per line (95% CI 0.22–0.28),
0.30 per scene (0.24–0.36). One-vs-rest AUC per partner:

| | Ross | Monica | Joey | Chandler | Phoebe | Other |
|---|---|---|---|---|---|---|
| line | 0.54 | 0.57 | 0.59 | 0.63 | 0.57 | 0.63 |
| scene | 0.56 | 0.61 | 0.61 | 0.73 | 0.62 | 0.72 |

Rachel's word choice alone only weakly identifies her partner. "Other" (dates, family) and Chandler are the
most distinctive; Ross is the weakest. For each partner X, the **ceiling** for a generator is
`gap(X) = mean P(X) on real lines to X − mean P(X) on real lines to others`. It's in
`partner_tfidf_metrics.json` with CIs. With TF-IDF, every partner's line-level gap except Ross's has a CI above
zero. A partner with a zero gap can't be evaluated this way.

## Tinker classifiers (persistent checkpoints)

`scripts/train_tinker.py` LoRA-fine-tunes `Qwen/Qwen3-8B` on Tinker's GPUs. The prompt lists the options as
letters and ends in `Answer:`; loss is only on the answer letter (class-weighted). Scoring is one prefill per
line: the sampler returns log-probs for every option letter at the answer position (`target_prompt_logprobs`),
renormalized over the options.

| | `rachel` | `partner` | why |
|---|---|---|---|
| LoRA rank | 16 | 8 | partner has ~7.5k train lines; lower capacity overfits less |
| LR | 2e-4 | 1e-4 | |
| epochs × batch | 2 × 128 | 3 × 32 | best epoch kept by val |
| val selection | AUC | mean one-vs-rest AUC | |
| cost (Qwen3-8B prices) | ~$1.50 | ~$0.85 | |

Before training, the **untrained** model is scored on val with the same prompt. If it is well above chance, it
is recognizing memorized Friends lines, and its test score will overstate how well it judges generated text.

**Checkpoints.** A resumable `save_state` every 50 steps (30-day TTL), a sampler checkpoint per epoch, the best
epoch's LoRA adapter exported in HF format and downloaded to `models/tinker/<run>/` (gitignored; loadable with
`peft` on any GPU), and a final state with no expiry. Paths are logged to
`results/classifiers/tinker_runs/<run>.jsonl` as they're written. The best checkpoint per task goes in
`results/classifiers/tinker_checkpoints.json`. Commit it so the team can load the classifiers by name:

```python
from friends.tinker_clf import TinkerClassifier
clf = TinkerClassifier("partner")                 # reads the registry
clf.predict_proba(["Oh. My. God."])               # [N, 6], columns = clf.classes_
```

```bash
pip install -r requirements.txt && export TINKER_API_KEY=...
python scripts/train_tinker.py --dry-run          # token counts, no Tinker calls
bash scripts/run_tinker_all.sh                    # list models, train partner + rachel, score Mistral
# crashed? copy the --resume line from results/classifiers/tinker_runs/<run>.jsonl
```

Offline tests (fake clients, real `tinker.types`): `TOKENIZER_DIR=<any HF tokenizer dir> pytest tests/`.

## Reproduce

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python scripts/build_classifier_data.py           # downloads raw JSON to data/raw/
python scripts/train_tfidf.py --task rachel
python scripts/train_tfidf.py --task partner
python scripts/score_generations.py               # judge the Mistral baseline with TF-IDF
python scripts/score_generations.py --rachel-model tinker --partner-model tinker \
    --out results/classifiers/baseline_generation_scores_tinker.json

# optional, any GPU
python scripts/train_transformer.py --task rachel --model roberta-base
python scripts/train_transformer.py --task partner --model roberta-base --epochs 6
```
