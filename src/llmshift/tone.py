"""Deterministic tone analysis.

A tone profile is a small set of interpretable markers (emoji, exclamation,
formality, hedging, apologies, follow-up questions, sentence length, voice...).
Comparing two profiles yields plain-English differences such as
"more casual" or "adds emoji", plus prompt instructions that undo them.
"""

from __future__ import annotations

import re

from .textutil import words

# Emoji blocks by code point. Built from numbers, not written as one literal class: CodeQL
# reads characters beyond U+FFFF as U+FFFD and then reports the ranges as overlapping.
_EMOJI_BLOCKS = ((0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF), (0x1FA70, 0x1FAFF), (0x2600, 0x26FF), (0x2700, 0x27BF))
EMOJI_RE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _EMOJI_BLOCKS) + "\u2b50\u2b55]")
# Typographic symbols in those blocks that structured answers use as bullets/marks.
_NOT_EMOJI = set("✓✔✗✘★☆➔➜➡❖●•☑☒☐☛☞")


def find_emoji(text: str) -> list[str]:
    return [ch for ch in EMOJI_RE.findall(text or "") if ch not in _NOT_EMOJI]

_INFORMAL = {
    "hey", "hiya", "yeah", "yep", "yup", "nope", "nah", "gonna", "wanna", "gotta", "kinda", "sorta", "cool", "awesome",
    "super", "totally", "lol", "btw", "omg", "stuff", "guys", "folks", "dude", "yay", "oops", "whoa", "wow",
    "cheers", "sweet", "ya", "y'all", "tons", "basically", "literally", "huge", "fab", "fantastic", "amazing", "pumped",
    "thrilled", "ok", "okay", "alright", "aww", "haha", "no worries",
}
_FORMAL = {
    "therefore", "furthermore", "moreover", "regarding", "accordingly", "consequently", "additionally",
    "nevertheless", "hence", "thus", "herein", "pursuant", "kindly", "sincerely", "respectfully", "whereas",
    "subsequently", "inquiry", "enquiry", "regards", "assistance", "appropriate", "ensure", "aforementioned",
    "please note", "we regret", "in accordance", "do not hesitate", "at your earliest convenience",
}
_HEDGES = {
    "might", "may", "perhaps", "possibly", "probably", "likely", "seems", "seem", "appears", "suggests",
    "somewhat", "arguably", "presumably", "i think", "i believe", "it is possible", "it's possible", "not sure",
    "i'm not certain", "could be", "it depends", "generally", "typically",
}
_APOLOGY = {"sorry", "apologize", "apologise", "apologies", "apology", "regret", "unfortunately", "apologizing", "apologising"}
_POLITE = {"please", "thank", "thanks", "appreciate", "happy to help", "glad to help", "you're welcome", "my pleasure", "kindly"}
_POSITIVE = {
    "great", "good", "excellent", "happy", "glad", "love", "wonderful", "fantastic", "amazing", "awesome", "perfect",
    "best", "delighted", "pleased", "excited", "enjoy", "nice", "brilliant", "success", "successful", "benefit",
    "easy", "helpful", "positive", "fortunately", "congratulations", "welcome", "thrilled", "superb", "ideal",
}
_NEGATIVE = {
    "bad", "poor", "terrible", "awful", "sad", "angry", "hate", "worst", "problem", "issue", "fail", "failed",
    "failure", "unfortunately", "wrong", "error", "difficult", "concern", "concerned", "disappointed", "frustrating",
    "negative", "risk", "danger", "dangerous", "cannot", "unable", "denied", "reject", "rejected", "broken", "delay",
}
_FOLLOW_UP_RE = re.compile(
    r"(?i)(would you like|do you want|shall i|should i|want me to|let me know if|if you(?:'d| would) like|"
    r"anything else|can i help|feel free to (?:ask|reach)|is there anything)"
)

# Contractions, but not possessives: "it's", "we're", "don't" count; "the account's" does not.
_CONTRACTION_RE = re.compile(
    r"(?i)\b(?:[a-z]+['’](?:re|ve|ll|d|m|t)|(?:it|that|there|here|what|who|where|how|he|she|let|that)['’]s)\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _phrase_count(text_lc: str, tokens: list[str], vocab: set[str]) -> int:
    single = {w for w in vocab if " " not in w}
    multi = [w for w in vocab if " " in w]
    count = sum(1 for t in tokens if t in single)
    for phrase in multi:
        count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lc))
    return count


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?(```|$)", " ", text or "", flags=re.S)


def profile(text: str) -> dict:
    """Compute tone markers for a piece of text."""
    clean = _strip_code(text)
    text_lc = clean.lower().replace("’", "'")
    toks = [w.lower().replace("’", "'") for w in words(clean)]
    n_words = max(1, len(toks))
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(clean.strip()) if re.search(r"\w", s)]
    n_sent = max(1, len(sentences))
    per100 = 100.0 / n_words
    informal = _phrase_count(text_lc, toks, _INFORMAL)
    formal = _phrase_count(text_lc, toks, _FORMAL)
    contractions = len(_CONTRACTION_RE.findall(clean))
    exclaims = clean.count("!")
    emoji = find_emoji(clean)
    positive = _phrase_count(text_lc, toks, _POSITIVE)
    negative = _phrase_count(text_lc, toks, _NEGATIVE)
    # Formality: 0 = very casual, 1 = very formal. The smoothing term keeps a single
    # marker (one "!" in a short answer) from swinging the score.
    casual_signal = informal * 2 + contractions + exclaims * 1.5 + len(emoji) * 2
    formal_signal = formal * 2
    smoothing = max(6.0, n_words / 10.0)
    formality = 0.5 + 0.5 * (formal_signal - casual_signal) / (formal_signal + casual_signal + smoothing)
    return {
        "words": len(toks),
        "sentences": n_sent,
        "avg_sentence_words": round(len(toks) / n_sent, 2),
        "emoji": len(emoji),
        "emoji_examples": "".join(dict.fromkeys(emoji))[:6],
        "exclamations": exclaims,
        "questions": clean.count("?"),
        "follow_up": bool(_FOLLOW_UP_RE.search(clean[-400:])),
        "contractions_per100": round(contractions * per100, 2),
        "informal_per100": round(informal * per100, 2),
        "formal_per100": round(formal * per100, 2),
        "formality": round(max(0.0, min(1.0, formality)), 3),
        "hedges_per100": round(_phrase_count(text_lc, toks, _HEDGES) * per100, 2),
        "hedges": _phrase_count(text_lc, toks, _HEDGES),
        "apologies": _phrase_count(text_lc, toks, _APOLOGY),
        "politeness": _phrase_count(text_lc, toks, _POLITE),
        "sentiment": round((positive - negative) / (positive + negative + 2.0), 3),
        "we_voice": sum(1 for t in toks if t in ("we", "our", "us", "we're", "we'll", "we've")),
        "i_voice": sum(1 for t in toks if t in ("i", "my", "me", "i'm", "i'll", "i've", "i'd")),
        "caps_words": sum(1 for w in words(clean) if len(w) >= 3 and w.isupper() and w.isalpha()),
    }


MIN_WORDS = 12


def compare(base: dict, cand: dict, tolerance: float = 1.0) -> tuple[float, list[dict]]:
    """Compare two tone profiles. Returns (score 0..1, differences).

    Each difference is {"marker", "direction", "text"}. `tolerance` > 1 makes the
    comparison more lenient, < 1 stricter.
    """
    t = max(0.1, float(tolerance))
    diffs: list[dict] = []

    def add(marker: str, direction: str, text: str) -> None:
        diffs.append({"marker": marker, "direction": direction, "text": text})

    if base.get("emoji", 0) == 0 and cand.get("emoji", 0) >= 1:
        ex = cand.get("emoji_examples") or ""
        add("emoji", "more", f"adds emoji {ex}".strip())
    elif base.get("emoji", 0) >= 2 and cand.get("emoji", 0) == 0:
        add("emoji", "less", "drops the emoji the old model used")

    be = base.get("exclamations", 0) / max(1, base.get("sentences", 1))
    ce = cand.get("exclamations", 0) / max(1, cand.get("sentences", 1))
    if ce - be >= 0.25 * t and cand.get("exclamations", 0) - base.get("exclamations", 0) >= 2:
        add("exclamations", "more", f"more exclamatory ({cand.get('exclamations', 0)} '!' vs {base.get('exclamations', 0)})")
    elif be - ce >= 0.25 * t and base.get("exclamations", 0) - cand.get("exclamations", 0) >= 2:
        add("exclamations", "less", f"less enthusiastic ({cand.get('exclamations', 0)} '!' vs {base.get('exclamations', 0)})")

    df = cand.get("formality", 0.5) - base.get("formality", 0.5)
    if df <= -0.18 * t:
        add("formality", "less", f"more casual (formality {base.get('formality', 0):.2f} -> {cand.get('formality', 0):.2f})")
    elif df >= 0.18 * t:
        add("formality", "more", f"more formal (formality {base.get('formality', 0):.2f} -> {cand.get('formality', 0):.2f})")

    ds = cand.get("sentiment", 0) - base.get("sentiment", 0)
    if ds <= -0.45 * t:
        add("sentiment", "less", "more negative wording")
    elif ds >= 0.45 * t:
        add("sentiment", "more", "more positive/upbeat wording")

    if cand.get("hedges_per100", 0) - base.get("hedges_per100", 0) >= 2.0 * t and cand.get("hedges", 0) - base.get("hedges", 0) >= 2:
        add("hedging", "more", "hedges more (might / perhaps / possibly)")
    elif base.get("hedges_per100", 0) - cand.get("hedges_per100", 0) >= 2.0 * t and base.get("hedges", 0) - cand.get("hedges", 0) >= 2:
        add("hedging", "less", "more assertive, fewer hedges")

    if base.get("apologies", 0) == 0 and cand.get("apologies", 0) >= 1:
        add("apologies", "more", "adds apologies")
    elif base.get("apologies", 0) >= 1 and cand.get("apologies", 0) == 0:
        add("apologies", "less", "no longer apologises")

    if cand.get("follow_up") and not base.get("follow_up"):
        add("follow_up", "more", "ends with a follow-up question or offer")
    elif base.get("follow_up") and not cand.get("follow_up"):
        add("follow_up", "less", "drops the closing offer of further help")

    if base.get("politeness", 0) >= 1 and cand.get("politeness", 0) == 0:
        add("politeness", "less", "less polite (no please/thank you)")

    bs, cs = base.get("avg_sentence_words", 0), cand.get("avg_sentence_words", 0)
    if bs >= 4 and cs >= 4:
        ratio = cs / bs
        if ratio >= 1 + 0.6 * t:
            add("sentence_length", "more", f"longer sentences ({cs:.0f} vs {bs:.0f} words on average)")
        elif ratio <= 1 / (1 + 0.6 * t):
            add("sentence_length", "less", f"shorter sentences ({cs:.0f} vs {bs:.0f} words on average)")

    if base.get("we_voice", 0) >= 2 and base.get("i_voice", 0) == 0 and cand.get("i_voice", 0) >= 2 and cand.get("we_voice", 0) == 0:
        add("voice", "i", "switches from 'we' (company voice) to 'I'")
    elif base.get("i_voice", 0) >= 2 and base.get("we_voice", 0) == 0 and cand.get("we_voice", 0) >= 2 and cand.get("i_voice", 0) == 0:
        add("voice", "we", "switches from 'I' to 'we'")

    if cand.get("caps_words", 0) - base.get("caps_words", 0) >= 3:
        add("caps", "more", "uses more ALL-CAPS words")

    score = max(0.0, 1.0 - 0.2 * len(diffs))
    return score, diffs


_INSTRUCTIONS = {
    ("emoji", "more"): "Do not use emoji.",
    ("emoji", "less"): "Use an occasional emoji where it fits the tone.",
    ("exclamations", "more"): "Avoid exclamation marks; keep a calm, measured tone.",
    ("exclamations", "less"): "Keep an upbeat, enthusiastic tone.",
    ("formality", "less"): "Use a professional, formal tone. Avoid slang and casual phrases.",
    ("formality", "more"): "Use a friendly, conversational tone; contractions are fine.",
    ("sentiment", "less"): "Keep the wording positive and constructive.",
    ("sentiment", "more"): "Keep the wording neutral and factual rather than upbeat.",
    ("hedging", "more"): "Be direct and confident. Do not hedge with words like 'might', 'perhaps' or 'possibly' unless you are genuinely unsure.",
    ("hedging", "less"): "Acknowledge uncertainty where appropriate.",
    ("apologies", "more"): "Do not apologise unless something has actually gone wrong.",
    ("apologies", "less"): "Acknowledge the user's problem and apologise for any inconvenience.",
    ("follow_up", "more"): "Do not end with follow-up questions or offers of further help; finish once the request is answered.",
    ("follow_up", "less"): "End by offering further help.",
    ("politeness", "less"): "Be courteous: say please and thank you where natural.",
    ("sentence_length", "more"): "Use short, simple sentences.",
    ("sentence_length", "less"): "Write in complete, well-developed sentences.",
    ("voice", "i"): "Speak on behalf of the company using 'we', not 'I'.",
    ("voice", "we"): "Speak in the first person singular ('I').",
    ("caps", "more"): "Do not use ALL-CAPS for emphasis.",
}


def instructions_for(diffs: list[dict]) -> list[str]:
    out = []
    for d in diffs:
        text = _INSTRUCTIONS.get((d["marker"], d["direction"]))
        if text and text not in out:
            out.append(text)
    return out


def describe(p: dict) -> str:
    bits = []
    f = p.get("formality", 0.5)
    bits.append("formal" if f >= 0.62 else "casual" if f <= 0.38 else "neutral")
    if p.get("emoji"):
        bits.append("uses emoji")
    if p.get("exclamations", 0) >= 2:
        bits.append("exclamatory")
    if p.get("follow_up"):
        bits.append("offers follow-up")
    if p.get("apologies"):
        bits.append("apologetic")
    return ", ".join(bits)
