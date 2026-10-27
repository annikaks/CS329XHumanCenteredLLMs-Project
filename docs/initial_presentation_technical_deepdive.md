# Initial Presentation: Technical Deep Dive

Friends classifiers for judging generated Rachel dialogue · Oct 27, 2026 · Annika

Two judges score generated Rachel dialogue: one estimates whether a line is in Rachel's voice, the other which friend she is talking to. Each was built twice, as TF-IDF + logistic regression and as a LoRA-fine-tuned Qwen3-8B, and the stronger version of each is used to score generations.

## Overview

Each judge is a classifier trained on real *Friends* transcripts, then applied to generated text. The same judge also scores the real line at the same moment, which gives the ceiling a generator can be held to.

| Judge | Input | Output | Used for scoring |
| --- | --- | --- | --- |
| Rachel judge (`rachel`) | one line, names masked | P(line is Rachel's), binary Rachel vs Other | Qwen3-8B + LoRA |
| Partner judge (`partner`) | Rachel's line(s) only; the partner's turns are removed | a distribution over Ross, Monica, Chandler, Joey, Phoebe, Other | TF-IDF + LR for single lines; scene-level Qwen3-8B + LoRA for multi-line generations |

The partner judge never sees the partner's words. In generation experiments those words are real context, so including them would score the context instead of the generated line.

Evaluation compares two columns per metric. The real-line column measures the judge: those lines are genuinely Rachel's, so every miss is the classifier's. The generated-line column measures the generator under the same judge. A generator is judged against the real-line score, not against 100%.

## Data

Both tasks are built from all ten seasons by `scripts/build_classifier_data.py`, with one shared split.

**Source.** The [emorynlp/character-mining](https://github.com/emorynlp/character-mining) JSON that ConvoKit's `friends-corpus` was built from. Taking `speaker = speakers[0]` per utterance reproduces ConvoKit exactly: 9,331 / 9,161 / 8,498 lines for Rachel / Ross / Monica and the notebook's 481 / 295 Rachel–Ross / Rachel–Monica segments.

**Split, in order:**

1. Remove the 111 scenes in `friends_split.json["test_episode_ids"]`. They are reserved for generation experiments and never reach a classifier.
2. Split the remaining 236 episodes (`sXX_eYY`) 80/20 with seed 42: 189 train, 47 test. Splitting by episode, not scene or line, keeps a storyline on one side, so topic words can't leak the label.
3. For model selection, the Tinker runs hold out a further 10% of training episodes as validation; TF-IDF uses 5-fold cross-validation grouped by episode.

**Name masking.** Every person's name becomes `[NAME]`:

- regex for the six mains and their nicknames (Rach, Mon, Phoebs, Joe, …), the notebook's `Person X` labels and `[Other Person]`
- spaCy `en_core_web_sm` PERSON entities, capitalized and longer than 2 characters, minus a stoplist of words it mistakes for names (*Y'know*, *Ya*, *Honey*, *Mom*, …)
- runs of adjacent `[NAME]` tokens collapsed to one

Without masking, "talking to Ross" is learnable from the word *Ross*, which measures nothing about style.

**Task examples.**

| Task | Example | Label | Train | Test |
| --- | --- | --- | --- | --- |
| `rachel` | one line by any of the six mains | Rachel / Other | 38,923 lines (6,823 Rachel) | 9,116 lines (1,520 Rachel) |
| `partner` | one Rachel line from a segment where only she and one partner speak | Ross, Monica, Chandler, Joey, Phoebe, Other | 8,319 lines in 4,208 segments | 1,056 lines in 287 segments |

A partner segment is a contiguous run of turns spoken only by Rachel and one partner. Training segments need at least 2 turns with at least 1 from each speaker; test segments keep the notebook's stricter rule of at least 4 turns with at least 2 each, so test labels are as clean as possible. On the same test set, the relaxed training rule raised line accuracy from 0.224 to 0.251 versus the strict rule's 4,836 training lines.

Partner test lines by class: Ross 230, Monica 147, Joey 250, Phoebe 156, Chandler 51, Other 222.

## Model 1: TF-IDF + logistic regression

A sparse bag-of-n-grams representation fed to a linear classifier (`scripts/train_tfidf.py`, scikit-learn). It trains on CPU in minutes and is the baseline both tasks must beat.

**Features.** Two TF-IDF vectorizers run side by side (`FeatureUnion`) and their outputs are concatenated:

| Block | Units | Settings | What it captures |
| --- | --- | --- | --- |
| word | word unigrams + bigrams | lowercased, min document frequency 2, sublinear tf | vocabulary and short phrases (*oh my*, *you know*) |
| char | character 2–5-grams within word boundaries (`char_wb`) | case kept, min document frequency 3, top 200,000 by frequency, sublinear tf | spelling, punctuation and casing habits (*Y'kn*, *!!*, *Ohh*) |

Each entry is a sublinear term frequency times a smoothed inverse document frequency, and each block's vector is L2-normalized:

$$
\mathrm{tfidf}(t,d) = \bigl(1 + \ln \mathrm{tf}(t,d)\bigr)\cdot\Bigl(\ln\tfrac{1+n}{1+\mathrm{df}(t)} + 1\Bigr)
$$

**Classifier.** L2-regularized logistic regression (lbfgs, up to 3,000 iterations). For the 6-way partner task it is the multinomial (softmax) form, one weight vector per class. Class weights are balanced, w_c = n / (K · n_c), so Chandler's 765 training lines count as much as Ross's 2,160:

$$
\min_{W,b}\; \tfrac{1}{2C}\lVert W\rVert_2^2 \;-\; \sum_{i} w_{y_i}\,\log \frac{\exp(W_{y_i}^\top x_i + b_{y_i})}{\sum_{k}\exp(W_k^\top x_i + b_k)}
$$

**Model selection.** `GridSearchCV` over C ∈ {0.1, 0.3, 1, 3} with 5-fold `GroupKFold` by episode on the training set only, scored by ROC AUC (`rachel`) or macro-F1 (`partner`), then refit on all training data. Selected: C = 0.1 for `rachel`, C = 1.0 for `partner`. The test split is scored once.

**Output.** `predict_proba` gives P(Rachel) or the 6 partner probabilities per line. Scene-level partner predictions sum the per-line log-probabilities over Rachel's lines in a segment, then renormalize with a softmax.

What it cannot model: word order beyond bigrams, meaning, or tone.

## Model 2: Qwen3-8B + LoRA on Tinker

A pretrained 8-billion-parameter language model, `Qwen/Qwen3-8B`, turned into a classifier by teaching it to answer a multiple-choice prompt with one letter (`scripts/train_tinker.py`, `src/friends/tinker_clf.py`). Training runs on Tinker's GPUs; locally there is only tokenization and bookkeeping.

**Prompt.** Each example is wrapped in a fixed template listing the options as letters. The partner task's prompt:

```text
This is something Rachel says in a two-person conversation from Friends. Who is she talking to? Character names are masked as [NAME].

Options:
A) Chandler
B) Joey
C) Monica
D) someone outside the main six
E) Phoebe
F) Ross

Rachel:
Oh my God, [NAME], you are SO sweet.

Answer:
```

The `rachel` task uses the same layout with two options, *A) Rachel* and *B) someone else*. Letters rather than names are the targets because every option must be exactly one, distinct token: in Qwen's tokenizer " A" … " O" are single tokens (" A" = id 362), which the script asserts at startup. Text longer than 1,024 tokens is truncated.

**Training example and loss.** For prompt tokens p₁…pₙ and answer letter a, the sequence is p₁…pₙa. The model input is the sequence minus its last token, the targets are the sequence shifted by one, and every target weight is 0 except the answer position, which gets the class weight w_c = n / (K · n_c). Tinker's `cross_entropy` loss sums weighted token losses, so per batch:

$$
\mathcal{L} = \sum_{i \in \text{batch}} w_{y_i}\,\bigl(-\log p_\theta(a_{y_i} \mid \text{prompt}_i)\bigr)
$$

The softmax here is over the full 151k-token vocabulary, so the model learns both the answer format and the label.

**LoRA.** The 8B base weights stay frozen. Each adapted weight matrix W gets a trainable low-rank update W + BA, with B of shape d×r and A of shape r×k, on the attention, MLP and unembedding layers (Tinker's defaults). Rank r is 16 for `rachel` and 8 or 16 for `partner`. This trains well under 1% of the parameters, which is cheaper and harder to overfit on a few thousand examples.

**Optimization.** Each step is a `forward_backward` then an `optim_step`. Adam with β₁ 0.9, β₂ 0.95, ε 1e-12, no weight decay and no gradient clipping (Tinker's defaults). The learning rate decays linearly from its peak to 0 over all steps, with no warmup. Training order is reshuffled each epoch with seed 42, which also makes resumed runs skip exactly the batches already seen.

**Model selection and the memorization check.** Before any training, the untrained base model is scored on validation with the same prompt. A score far above chance would mean it recognizes memorized *Friends* lines, inflating test scores relative to generated text. After each epoch a sampler checkpoint is saved and scored on validation: AUC of P(Rachel) for `rachel`, mean one-vs-rest AUC for `partner`. The best epoch is scored once on test.

**Scoring (inference).** One request per example, no sampling. The prompt is sent with a placeholder answer token appended, and the sampler is asked for the log-probability of each option letter at the answer position (`target_prompt_logprobs`). Those K log-probabilities are renormalized with a softmax over the options only, giving the class probabilities.

**Scene-level variant (`--unit segment`, partner only).** Instead of one Rachel line, the example is all of Rachel's lines in a segment joined by newlines, one example per segment: 3.8k training examples instead of 7.5k, each with far more text.

**Checkpoints.** A resumable `save_state` (weights + optimizer state) every 50 steps with a 30-day TTL, a sampler checkpoint per epoch, the best epoch exported in Hugging Face PEFT format and downloaded to `models/tinker/<run>/`, and a final state with no expiry. The best sampler path per task is recorded in `results/classifiers/tinker_checkpoints.json`, so `TinkerClassifier("rachel")` loads it by name.

## Training runs

Three Tinker runs on Qwen3-8B, about $3 of credit in total; the best epoch of each was kept.

| Run | Unit | LoRA rank | Peak LR | Epochs × batch | Steps | Val: untrained → ep 0 → ep 1 → ep 2 | Approx. cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rachel` | line | 16 | 2e-4 | 2 × 128 | 554 | 0.548 → 0.636 → **0.668** | $1.50 |
| `partner` v1 | line | 8 | 1e-4 | 3 × 32 | 711 | 0.536 → 0.533 → 0.561 → **0.587** | $0.85 |
| `partner` v2 (in use) | segment | 16 | 4e-4 | 3 × 32 | ~360 | 0.536 → 0.496 → 0.563 → **0.584** | $0.50 |

Validation scores are AUC for `rachel` and mean one-vs-rest AUC for `partner`; chance is 0.5 for both.

Why these settings: `rachel` has about 35k short lines, enough for rank 16 and two passes. `partner` has a tenth of that, so v1 used a lower rank and learning rate against overfitting. Its training loss stayed near ln 6 ≈ 1.79 (uniform guessing over 6 options) for most of epoch 0 and validation was still rising at the end, so v2 raised the learning rate and gave the model whole segments to read.

Both untrained scores sit near 0.5, so Qwen3-8B does not recognize these lines from pretraining, and the test scores below are not inflated by memorization.

## Metrics

All metrics are computed by `src/friends/evaluation.py`, identically for every model.

- **AUC (`rachel`).** The probability that a random Rachel line gets a higher P(Rachel) than a random non-Rachel line; 0.5 is chance, 1.0 perfect. Used instead of accuracy because only 17% of lines are Rachel's, so always answering "Other" scores 83% accuracy.
- **One-vs-rest AUC (`partner`).** The same, per partner: lines to X scored by P(X) against lines to everyone else. The headline is the mean over the six classes.
- **Partner gap.** For partner X: mean P(X) on real lines to X minus mean P(X) on real lines to anyone else. This is the ceiling on how much partner-specific signal a generator can show under that judge; a gap of zero means partner X cannot be evaluated this way.
- **Aggregating lines.** Several lines from one speaker in one scene are combined by averaging log-odds (`rachel`) or summing log-probabilities and renormalizing (`partner`). This mirrors judging a multi-line generation.
- **Confidence intervals.** 95% intervals are cluster bootstraps over scenes (1,000 resamples): whole scenes are resampled, because lines from one scene are not independent. Treating lines as independent would make the intervals too narrow.
- **Generation scores.** A line counts as Rachel's when P(Rachel) > 0.5. The partner call is either the higher of P(Ross) and P(Monica) (chance 50%) or the top class among all six (chance about 17%). Paired bootstrap intervals compare real and generated lines at the same moments.

## Classifier test results

Qwen3-8B is the better Rachel judge; for the partner task the two models tie at scene level, with different strengths by partner. All numbers are on the 47 held-out test episodes.

**`rachel`** (9,116 lines, AUC, 95% CI in brackets):

| Lines scored | Qwen3-8B + LoRA | TF-IDF + LR |
| --- | --- | --- |
| 1 line | **0.677** [0.661, 0.691] | 0.660 [0.644, 0.676] |
| 1 line, 1–3 words | 0.635 | 0.640 |
| 1 line, 16+ words | **0.772** | 0.730 |
| 3 lines averaged | **0.789** | 0.766 |
| 5 lines averaged | **0.824** | 0.791 |
| 10 lines averaged | **0.898** | 0.847 |

The models tie on very short lines, which carry almost no signal (*Okay.*, *What?*). Qwen's lead grows with line length and when lines are combined, which is how generations are judged.

**`partner`** (one-vs-rest AUC; 1,056 lines in 287 scenes):

| Partner | TF-IDF, line | Qwen v1, line | TF-IDF, scene | Qwen v1, lines combined per scene | Qwen v2, scene input |
| --- | --- | --- | --- | --- | --- |
| Ross | 0.541 | 0.557 | 0.558 | 0.578 | 0.571 |
| Monica | 0.571 | 0.546 | 0.613 | 0.586 | 0.638 |
| Joey | 0.591 | 0.511 | 0.607 | 0.512 | 0.556 |
| Chandler | 0.626 | 0.498 | 0.733 | 0.519 | 0.663 |
| Phoebe | 0.569 | 0.621 | 0.623 | 0.680 | 0.624 |
| Other | 0.629 | 0.710 | 0.722 | 0.825 | 0.762 |
| **Mean** | **0.588** | 0.574 | **0.643** | 0.617 | **0.636** |
| Accuracy (6-way) | 0.251 | 0.210 | 0.300 | 0.268 | 0.268 |

The most-common-partner baseline is 0.237 per line and 0.202 per scene. Per-partner intervals are about ±0.08 at scene level, so TF-IDF (0.643) and Qwen v2 (0.636) are statistically tied. Rachel's words alone only weakly reveal her partner: "Other" (dates, family) is the most distinctive, Ross the least.

## Scoring the Mistral pilot

Mistral-7B-Instruct's lines are judged Rachel's 25% of the time against 65% for her real lines, and they track Monica worse than the real lines do; the second finding needs more data to confirm.

The pilot (`BaselineRun.ipynb`) took 40 moments from the held-out scenes, 20 with Ross and 20 with Monica. At each, few-shot Mistral generated Rachel's next line from up to 6 previous turns; the real next line is the comparison. `scripts/score_generations.py` masks names and scores each line on its own, and `scripts/final_report.py` writes the per-line table to `results/classifiers/final_report.csv`.

**Table 1: is this line Rachel's?** (P > 0.5)

| Judge | Real Rachel lines | Mistral's lines |
| --- | --- | --- |
| Qwen3-8B + LoRA (used) | 26/40 (65%) | 10/40 (25%) |
| TF-IDF + LR | 19/40 (48%) | 4/40 (10%) |

With the Qwen judge, the mean P(Rachel) is 0.56 for real lines and 0.38 for Mistral's, a difference of 0.18 (paired 95% CI 0.12 to 0.24); Mistral's line scores lower at 34 of 40 moments.

**Table 2: who is Rachel talking to?** (TF-IDF partner judge, single lines)

| | Real Rachel lines | Mistral's lines |
| --- | --- | --- |
| Ross scenes → Ross (Ross vs Monica) | 14/20 (70%) | 13/20 (65%) |
| Monica scenes → Monica (Ross vs Monica) | 13/20 (65%) | 9/20 (45%) |
| Ross scenes → Ross (all 6 partners) | 9/20 (45%) | 2/20 (10%) |
| Monica scenes → Monica (all 6 partners) | 7/20 (35%) | 4/20 (20%) |

Mistral's lines lean "Ross" whoever Rachel is talking to: 24 of 40 are called Ross, including 11 of the 20 Monica scenes. The real-minus-generated difference on Ross vs Monica is 0.12, but its paired interval (−0.07 to 0.33) includes zero.

The scene-level Qwen partner judge is not used here: every pilot generation is a single line, and given one line it scores real lines at chance (0.53 on Ross vs Monica). It is for multi-line generations, with all of Rachel's generated lines passed together.

## Limitations and next steps

- **Small pilot.** 20 lines per partner moves each percentage 5 points per line. Running `BaselineRun.ipynb` with `N_TEST_PER_PARTNER = None` gives all 535 test moments (373 Ross, 162 Monica) and roughly ±0.08 intervals.
- **Weak partner signal.** Both partner judges reach only about 0.64 mean AUC at scene level, so differences between generators must be large to register. Averaging the TF-IDF and Qwen partner probabilities may beat both, since they are strong on different partners.
- **Segment noise.** "Only two people speak" does not mean only two are present; a silent third character can be the real addressee.
- **Generation format.** Some Mistral outputs contain extra turns ("Person C: … Person A: …"), so the judges score several speakers at once. These should be stripped, or the prompt tightened, before the full run.
- **Calibration.** Class weighting shifts the probabilities, so compare probabilities between real and generated lines rather than reading them as absolute likelihoods.

## Reproduce

```bash
python scripts/build_classifier_data.py
python scripts/train_tfidf.py --task rachel
python scripts/train_tfidf.py --task partner
python scripts/train_tinker.py --task rachel
python scripts/train_tinker.py --task partner --unit segment --lr 4e-4 --rank 16 --epochs 3
python scripts/score_generations.py --results <generations.json> --rachel-model tinker --partner-model models/partner_tfidf.joblib --out <scores.json>
python scripts/final_report.py --results <generations.json> --scores <scores.json> --out <report>
```
