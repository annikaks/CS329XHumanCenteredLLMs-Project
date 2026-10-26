"""Replace person names with [NAME] so classifiers learn style, not who gets named.

Covers the six mains + nicknames, the notebook's "Person X" labels (so anonymized
data and model generations are treated the same as raw transcripts), and any other
PERSON entity spaCy finds.
"""
import re

MAIN_NAME_VARIANTS = [
    "rachel green", "rachel", "rach", "rachie", "rachem",
    "ross geller", "ross", "rossy", "rossie", "ross-a-tron",
    "monica geller", "monica", "mon", "monnie", "mon-mon",
    "chandler bing", "chandler", "chan", "chandie", "bing",
    "joey tribbiani", "joey", "joe", "joseph", "tribbiani",
    "phoebe buffay", "phoebe", "phoebs", "pheebs", "pheebz", "buffay",
    "geller", "gellers",
]
PLACEHOLDER = "[NAME]"

# spaCy tags these as PERSON in this corpus; most are speech-style markers worth keeping.
NER_STOPLIST = {
    "y'know", "ya know", "ya", "yknow", "nah", "wanna", "whoah", "ohh", "kinda", "whaddya",
    "tellin", "lemme", "nono", "mornin", "calm", "hurry", "honey", "sweetie", "miss",
    "dad", "mom", "mommy", "daddy", "grandma", "grandmother", "uncle", "aunt", "wha",
    "uh", "um", "umm", "oh", "god", "dude", "sweet", "back", "big", "la", "se", "sup",
    "allright", "okidoki", "valentine", "vegas", "jell", "peanut",
}

_main_re = re.compile(
    r"(?<![\w-])(?<!\w['’])(" +"|".join(re.escape(n) for n in sorted(MAIN_NAME_VARIANTS, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)
_person_label_re = re.compile(r"\bPerson [A-Z]{1,2}\b")
_other_person_re = re.compile(r"\[Other Person\]")
_collapse_re = re.compile(r"(\[NAME\](?:[\s,]*\[NAME\])+)")

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser", "tagger"])
    return _nlp


def _regex_mask(text):
    text = _person_label_re.sub(PLACEHOLDER, text)
    text = _other_person_re.sub(PLACEHOLDER, text)
    return _main_re.sub(PLACEHOLDER, text)


def _ner_mask(text, doc):
    out, last = [], 0
    for ent in doc.ents:
        name = ent.text.strip()
        if (
            ent.label_ == "PERSON"
            and PLACEHOLDER not in name
            and len(name) > 2
            and name[0].isupper()
            and name.lower() not in NER_STOPLIST
        ):
            out.append(text[last:ent.start_char])
            out.append(PLACEHOLDER)
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


def mask_names(texts, use_ner=True, batch_size=256):
    """Mask a list of strings. Returns a list of the same length."""
    texts = [_regex_mask(t or "") for t in texts]
    if use_ner:
        nlp = _get_nlp()
        texts = [_ner_mask(t, d) for t, d in zip(texts, nlp.pipe(texts, batch_size=batch_size))]
    return [_collapse_re.sub(PLACEHOLDER, t) for t in texts]
