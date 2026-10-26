"""Load the Friends corpus and extract per-character utterances / pair-only segments.

Two loaders produce the same structure: a list of (conversation_id, [(speaker, text), ...]).
- ConvoKit (what the data notebook uses; needs zissou.infosci.cornell.edu)
- emorynlp/character-mining JSON (the source ConvoKit's friends-corpus was built from;
  speaker = speakers[0], matching ConvoKit — reproduces the 481 A-B / 295 A-C segments)
"""
import glob
import json
import os
import urllib.request
from collections import Counter
from itertools import combinations

MAIN_SIX = [
    "Rachel Green",
    "Ross Geller",
    "Monica Geller",
    "Chandler Bing",
    "Joey Tribbiani",
    "Phoebe Buffay",
]
ALL_PAIRS = [tuple(sorted(p)) for p in combinations(MAIN_SIX, 2)]

EMORY_URL = "https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_{:02d}.json"


def download_emory(raw_dir="data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    for s in range(1, 11):
        path = os.path.join(raw_dir, f"friends_season_{s:02d}.json")
        if not os.path.exists(path):
            urllib.request.urlretrieve(EMORY_URL.format(s), path)
    return raw_dir


def load_conversations_emory(raw_dir="data/raw"):
    convos = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "friends_season_*.json"))):
        with open(f, encoding="utf-8") as fh:
            season = json.load(fh)
        for ep in season["episodes"]:
            for scene in ep["scenes"]:
                utts = scene["utterances"]
                if not utts:
                    continue
                turns = [
                    (u["speakers"][0] if u["speakers"] else "NONE", u["transcript"])
                    for u in utts
                ]
                convos.append((utts[0]["utterance_id"], turns))
    return convos


def load_conversations_convokit():
    from convokit import Corpus, download

    corpus = Corpus(filename=download("friends-corpus"))
    return [
        (c.id, [(u.speaker.id, u.text) for u in c.iter_utterances()])
        for c in corpus.iter_conversations()
    ]


def load_conversations(source="emory", raw_dir="data/raw"):
    if source == "convokit":
        return load_conversations_convokit()
    download_emory(raw_dir)
    return load_conversations_emory(raw_dir)


def extract_pair_segments(convos, pair, min_turns=4, min_turns_per_person=2):
    """Same logic as 329XData.ipynb: contiguous runs where only `pair` speaks."""
    pair = set(pair)
    segments = []
    for cid, turns in convos:
        current = []

        def save():
            if len(current) < min_turns:
                return
            counts = Counter(sp for sp, _ in current)
            if any(counts[p] < min_turns_per_person for p in pair):
                return
            segments.append({
                "conversation_id": cid,
                "pair": sorted(pair),
                "turns": [{"speaker": sp, "text": t} for sp, t in current],
            })

        for sp, text in turns:
            if sp in pair:
                current.append((sp, text))
            else:
                save()
                current = []
        save()
    return segments


def episode_of(conversation_id):
    """s01_e05_c12_u001 -> s01_e05"""
    return conversation_id[:7]
