# CS 329X: Partner-conditioned dialogue on *Friends*

Does a model's imitation of a character (Person A = Rachel) change depending on who she's talking to
(Person B = Ross vs. Person C = Monica)? This repo has the data pipeline, the few-shot Mistral baseline, and
two classifiers used as automatic judges of generated dialogue.

## Layout

```
notebooks/
  329XData.ipynb        ConvoKit -> Rachel–Ross / Rachel–Monica pair segments, anonymization, scene-level split
  BaselineRun.ipynb     Mistral-7B-Instruct few-shot generation (pilot: 20 A–B + 20 A–C test examples)
data/processed/
  friends_AB_anonymized.json, friends_AC_anonymized.json   481 / 295 segments
  friends_split.json    80/20 split of scene ids; test_episode_ids = HELD OUT for generation testing
data/classifiers/       built by scripts/build_classifier_data.py (see below)
results/baseline/       Mistral outputs
results/classifiers/    metrics JSON + confusion matrices
src/friends/            corpus loading, segment extraction, name masking
scripts/                build data, train TF-IDF, fine-tune transformer, score generations
```

## Classifiers

| | Task 1: speaker | Task 2: pair |
|---|---|---|
| Input | one utterance | a pair-only segment, speakers relabeled `S1`/`S2` |
| Labels | 6 mains | all 15 pairs of the 6 mains |
| Train / test | 38,923 / 9,116 utterances | 2,771 / 659 segments |

**Data decisions**
- **Held-out scenes are excluded entirely.** Every scene in `friends_split.json["test_episode_ids"]` (111 scenes)
  is dropped before anything else, so neither classifier has seen the scenes we generate from.
- **Train/test is split by episode** (`sXX_eYY`), shared by both tasks. Splitting by utterance or scene would put
  lines from the same storyline on both sides and inflate test accuracy.
- **Names are masked to `[NAME]`.** This covers the six mains plus nicknames, the notebook's `Person X` labels, `[Other Person]`, and spaCy
  PERSON entities (with a stoplist so style markers like *Y'know* survive). Without masking, the pair classifier
  learns "the segment mentions Ross" and the speaker classifier learns who gets addressed by name. That's not
  style, and the generations are anonymized anyway.
- Segments use the notebook's exact rule (contiguous turns with only the pair speaking, ≥4 turns, ≥2 each).
  Rachel–Ross and Rachel–Monica training counts match the notebook (383 / 240).

**Source data.** `329XData.ipynb` uses ConvoKit's `friends-corpus`. The scripts default to the
[emorynlp/character-mining](https://github.com/emorynlp/character-mining) JSON that ConvoKit's corpus was built
from (`speaker = speakers[0]`). This reproduces ConvoKit's per-character utterance counts and the 481/295
segments exactly. Use `--source convokit` to read ConvoKit directly.

## Results (TF-IDF word+char n-grams + logistic regression, C picked by episode-grouped 5-fold CV)

| Task | Test acc | Macro-F1 | Chance |
|---|---|---|---|
| Speaker (6-way, single utterance) | 0.291 | 0.289 | 0.167 |
| Speaker, avg. log-prob over 3 / 5 / 10 utterances | 0.39 / 0.46 / 0.55 | | 0.167 |
| Pair (15-way, segment) | 0.208 | 0.172 | 0.067 |

Speaker accuracy by utterance length: 1–3 words 0.25, 4–7 0.25, 8–15 0.30, 16+ 0.37.
Single lines like "Okay." carry almost no signal, so treat the speaker judge as a corpus-level metric
(averaged over many generations), not a per-line verdict.

**Mistral baseline scored by these judges** (`results/classifiers/baseline_generation_scores.json`, n=40, so this is noisy):

| | gold response | generated |
|---|---|---|
| Speaker top-1 = Rachel | 0.25 | 0.075 |
| Mean P(Rachel) | 0.187 | 0.138 |
| Pair acc, Rachel–Ross vs Rachel–Monica (context + response) | 0.70 | 0.625 |

## Reproduce

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python scripts/build_classifier_data.py                 # downloads raw JSON to data/raw/
python scripts/train_tfidf.py --task speaker
python scripts/train_tfidf.py --task pair
python scripts/score_generations.py                     # judge the Mistral baseline outputs

# Colab GPU
python scripts/train_transformer.py --task speaker --model roberta-base --epochs 3
python scripts/train_transformer.py --task pair --model roberta-base --epochs 6 --batch-size 16
```
