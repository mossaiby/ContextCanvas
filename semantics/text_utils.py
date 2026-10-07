"""Dependency-free text utilities shared by the schema, the Context Graph and the pipeline.

Everything here is deterministic string processing so it can be unit-tested without
an LLM, KùzuDB or NetworkX. The functions deliberately avoid dataset-specific rules:
they encode general properties of names, dates, places and kinship phrases.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Possessives, folding, slugs
# --------------------------------------------------------------------------------------

_POSSESSIVE_RE = re.compile(r"(?<=\w)['’]s\b")
_PLURAL_POSSESSIVE_RE = re.compile(r"(?<=s)['’](?=\s|$)")


def strip_possessives(text: str) -> str:
    """Removes English possessive markers ("Julia's House" -> "Julia House", "Jones'" -> "Jones")."""
    text = _POSSESSIVE_RE.sub("", text)
    return _PLURAL_POSSESSIVE_RE.sub("", text)


def fold(text: str) -> str:
    """Case-folds and strips diacritics for comparison purposes."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def slugify(key: str, max_len: int = 60) -> str:
    """Unicode-aware slug. Falls back to a hash so non-Latin names never collapse to one id."""
    slug = re.sub(r"[^\w]+", "_", fold(key), flags=re.UNICODE).strip("_")
    if not slug:
        return "h" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_") + "_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]
    return slug


# --------------------------------------------------------------------------------------
# Entity surface normalisation and matching keys
# --------------------------------------------------------------------------------------

_LEADING_PREPOSITION_RE = re.compile(r"^(?:with|by|from|to|at|for|in|on|as)\s+", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_LEADING_PRONOUN_RE = re.compile(r"^(?:his|her|their|its)\s+", re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Occupational / role descriptors that commonly precede a proper name ("guitarist Grant Green").
# They are only stripped when written in lower case AND followed by a capitalised token, so
# proper names that merely start with such a word ("Producer Records") are left intact.
# Relational phrases with "of" ("son of X", "wife of X") are NOT stripped: they carry meaning
# and are converted into explicit relation events by the pipeline instead.
_DESCRIPTOR_WORDS = (
    "actor|actress|musician|guitarist|singer|songwriter|author|writer|novelist|poet|painter|"
    "director|producer|composer|journalist|politician|businessman|businesswoman|entrepreneur|"
    "footballer|player|coach|rapper|comedian|architect|scientist|physicist|chemist|engineer"
)
_DESCRIPTOR_RE = re.compile(r"^(?:" + _DESCRIPTOR_WORDS + r")\s+(?=[A-Z])")

_ADMIN_INVERSION_RE = re.compile(
    r"^(municipality|government|city|state|province|county|district|borough|town)\s+of\s+(.+)$",
    re.IGNORECASE,
)

# Honorifics only participate in the matching key (never in the display name), and only when
# at least two tokens remain, so "King Crimson" or "Queen Victoria" are not reduced.
_HONORIFICS = {"sir", "dame", "dr", "mr", "mrs", "ms", "lord", "lady", "king", "queen", "rev", "reverend", "professor", "prof"}

# Legal / club suffixes ignored in the matching key when at least one other token remains.
_ENTITY_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "plc",
    "gmbh", "ag", "sa", "nv", "bv", "fc", "afc", "cf", "sc",
}


_ABBREVIATIONS = {"inc", "corp", "co", "ltd", "jr", "sr", "st", "dr", "bros", "no", "mr", "mrs", "ms", "u.s", "d.c"}


def _strip_sentence_period(text: str) -> str:
    """Drops a sentence-final period but keeps it on abbreviations ("Inc.", "Jr.", "F.C.")."""
    if not text.endswith("."):
        return text
    last = text[:-1].split()[-1] if text[:-1].split() else ""
    if not last or "." in last or last.lower() in _ABBREVIATIONS or len(last) <= 2:
        return text
    return text[:-1].rstrip()


def normalize_surface(name: str) -> str:
    """Produces the display form of an entity mention (light, meaning-preserving cleanup)."""
    original = (name or "").strip()
    clean = original.strip("\"'“”‘’,;:").strip()
    clean = _strip_sentence_period(clean)
    clean = _LEADING_PREPOSITION_RE.sub("", clean).strip()
    clean = re.sub(r"['’]s$", "", clean).strip()
    clean = _LEADING_ARTICLE_RE.sub("", clean).strip()
    clean = _LEADING_PRONOUN_RE.sub("", clean).strip()
    clean = _DESCRIPTOR_RE.sub("", clean).strip()
    clean = _TRAILING_PAREN_RE.sub("", clean).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean if clean else original


def entity_match_key(name: str) -> str:
    """Canonical key used for entity resolution. Stricter equivalence than display identity."""
    surface = normalize_surface(name)
    m = _ADMIN_INVERSION_RE.match(surface)
    if m:
        surface = f"{m.group(2).strip()} {m.group(1).strip()}"
    tokens = [t for t in re.split(r"[^\w]+", fold(surface), flags=re.UNICODE) if t]
    tokens = [t for t in tokens if t not in {"the", "a", "an", "of", "and"}] or tokens
    if len(tokens) >= 3 and tokens[0] in _HONORIFICS:
        tokens = tokens[1:]
    while len(tokens) >= 2 and tokens[-1] in _ENTITY_SUFFIXES:
        tokens = tokens[:-1]
    # "F.C." becomes ["f", "c"]; collapse runs of single letters so it matches "FC".
    collapsed: List[str] = []
    buffer = ""
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buffer += t
            continue
        if buffer:
            collapsed.append(buffer)
            buffer = ""
        collapsed.append(t)
    if buffer:
        collapsed.append(buffer)
    while len(collapsed) >= 2 and collapsed[-1] in _ENTITY_SUFFIXES:
        collapsed = collapsed[:-1]
    return " ".join(collapsed)


# --------------------------------------------------------------------------------------
# Dates, numbers, places
# --------------------------------------------------------------------------------------

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
}
_DATE_VOCAB = _MONTHS | {
    "bc", "bce", "ad", "ce", "c", "circa", "ca", "early", "late", "mid", "spring", "summer",
    "autumn", "fall", "winter", "season", "from", "to", "and", "or", "between", "since", "until",
    "till", "in", "on", "of", "the", "around", "about", "approximately", "year", "years", "decade",
    "century", "centuries", "s", "st", "nd", "rd", "th", "annually", "born", "died",
}
_YEAR_RE = re.compile(r"^(1\d{3}|20\d{2})$")


def is_date_like(value: str) -> bool:
    """True when a string is a pure temporal expression ("May 4, 1948", "2010–11 season")."""
    tokens = re.findall(r"[A-Za-z]+|\d+", value or "")
    if not tokens:
        return False
    has_anchor = False
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if tok.isdigit():
            if _YEAR_RE.match(tok):
                has_anchor = True
            elif i + 1 < len(tokens) and tokens[i + 1].lower() in {"bc", "bce", "ad", "ce"}:
                has_anchor = True
            continue
        if low in _MONTHS:
            has_anchor = True
        if low not in _DATE_VOCAB:
            return False
    return has_anchor


def is_numeric_quantity(value: str) -> bool:
    """Detects thousands-separated numbers so "28,556" is never split as "28" + "556"."""
    return bool(re.search(r"\d,\d{3}", value or ""))


_TITLE_WORDS = {
    "duke", "duchess", "prince", "princess", "earl", "count", "countess", "baron", "baroness",
    "lord", "lady", "king", "queen", "emperor", "empress", "marquis", "viscount", "sir", "jr", "sr",
}


def split_place(value: str, max_parts: int = 5) -> Optional[List[str]]:
    """Splits a comma-qualified place ("Canyon, Texas, United States") into its parts.

    Returns None when the string does not look like a hierarchical place name, e.g. when it
    contains digits (addresses, quantities), lower-case fragments ("..., and St. Mary's counties")
    or too many parts (lists of names).
    """
    if not value or "," not in value or is_numeric_quantity(value):
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) < 2 or len(parts) > max_parts or any(not p for p in parts):
        return None
    for p in parts:
        if p.split()[0].lower() in _TITLE_WORDS:
            return None
        if re.search(r"\d", p):
            return None
        if not (p[0].isupper() or p[0] in "ÁÉÍÓÚÀÈÌÒÙÄÖÜÇ"):
            return None
        if len(p.split()) > 5:
            return None
    return parts


# --------------------------------------------------------------------------------------
# Kinship phrases
# --------------------------------------------------------------------------------------

_KIN_NOUNS = {
    "son": "child", "sons": "child", "daughter": "child", "daughters": "child", "child": "child",
    "children": "child", "stepson": "child", "stepdaughter": "child",
    "father": "parent", "mother": "parent", "parent": "parent", "parents": "parent",
    "stepfather": "parent", "stepmother": "parent",
    "wife": "spouse", "husband": "spouse", "spouse": "spouse", "widow": "spouse", "widower": "spouse",
    "consort": "spouse",
    "brother": "sibling", "brothers": "sibling", "sister": "sibling", "sisters": "sibling",
    "sibling": "sibling", "siblings": "sibling", "half-brother": "sibling", "half-sister": "sibling",
}
_KIN_ALT = "|".join(sorted((re.escape(k) for k in _KIN_NOUNS), key=len, reverse=True))
_MODIFIERS = r"(?:(?:the|a|an|his|her|their|its|only)\s+)?(?:[a-z][a-z'-]*\s+){0,3}?"
_KIN_OF_RE = re.compile(r"^" + _MODIFIERS + r"(" + _KIN_ALT + r")\s+(?:of|to)\s+(.+)$")
_KIN_BARE_RE = re.compile(r"^" + _MODIFIERS + r"(" + _KIN_ALT + r")$")
_BORN_TO_RE = re.compile(r"^(?:born|borne)\s+to\s+(.+)$")


def _lower_first(value: str) -> str:
    value = value.strip()
    return value[:1].lower() + value[1:] if value else value


def split_conjoined_names(value: str) -> List[str]:
    """Splits "A and B" / "A, B and C" when every part looks like a proper name."""
    parts = [p.strip() for p in re.split(r",\s*|\s+and\s+|\s*&\s*", value) if p.strip()]
    if len(parts) > 1 and all(p[:1].isupper() for p in parts):
        return parts
    return [value.strip()]


_PLURAL_KIN = {"sons", "daughters", "children", "parents", "brothers", "sisters", "siblings"}


def parse_kinship_phrase(value: str) -> Optional[Tuple[str, List[str]]]:
    """Parses "elder sister of X", "son of X and Y", "born to X and Y".

    Returns (relation, [targets]) where relation describes the SUBJECT relative to the targets:
    'child' (subject is child of targets), 'parent', 'spouse' or 'sibling'.
    """
    text = _lower_first(value)
    m = _BORN_TO_RE.match(text)
    if m:
        return "child", split_conjoined_names(value.strip()[len(value.strip()) - len(m.group(1)):])
    m = _KIN_OF_RE.match(text)
    if not m or m.group(1) in _PLURAL_KIN:
        return None  # "children of X and Y" names a group, not a relation of the subject
    relation = _KIN_NOUNS[m.group(1)]
    target_raw = value.strip()[len(value.strip()) - len(m.group(2)):]
    if not target_raw[:1].isupper():
        return None
    return relation, split_conjoined_names(target_raw)


def parse_bare_kin_noun(value: str) -> Optional[str]:
    """Recognises a bare relational noun phrase ("younger son", "eldest daughter")."""
    m = _KIN_BARE_RE.match(_lower_first(value))
    if not m or m.group(1) in _PLURAL_KIN:
        return None
    return _KIN_NOUNS[m.group(1)]


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """Splits text into sentence-aligned chunks of at most ~max_chars characters."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------------------
# Question -> entity anchoring
# --------------------------------------------------------------------------------------

_FUNCTION_WORDS = {
    "what", "which", "where", "when", "whose", "who", "whom", "how", "why", "in", "on", "at", "of",
    "is", "was", "are", "were", "does", "did", "do", "the", "a", "an", "to", "for", "by", "from",
    "with", "and", "or",
}
_SPAN_CONNECTORS = {
    "of", "de", "de'", "di", "da", "del", "della", "von", "van", "der", "den", "la", "le", "du",
    "des", "the", "and", "&", "y", "al", "el", "bin", "ibn", "on", "upon",
}
_TOKEN_RE = re.compile(r"[^\s,?!;\"“”()]+")


def _is_capitalised(token: str) -> bool:
    return bool(token) and (token[0].isupper() or token[0].isdigit())


def capitalised_spans(question: str) -> List[Tuple[int, int, str]]:
    """Maximal runs of capitalised tokens (allowing name connectors inside), with offsets."""
    tokens = [(m.start(), m.end(), m.group(0).rstrip(".:")) for m in _TOKEN_RE.finditer(question)]
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        if not _is_capitalised(tokens[i][2]) or tokens[i][2].lower() in _FUNCTION_WORDS - {"the"}:
            i += 1
            continue
        j = i
        last_cap = i
        while j + 1 < len(tokens):
            nxt = tokens[j + 1][2]
            if _is_capitalised(nxt) and nxt.lower() not in _FUNCTION_WORDS - {"the"}:
                j += 1
                last_cap = j
            elif nxt.lower() in _SPAN_CONNECTORS:
                j += 1
            else:
                break
        start, end = tokens[i][0], tokens[last_cap][0] + len(tokens[last_cap][2])
        spans.append((start, end, question[start:end]))
        i = last_cap + 1
    return spans


def _name_variants(name: str) -> List[str]:
    base = strip_possessives(name).strip()
    variants = [base]
    no_paren = re.sub(r"\s*\([^)]*\)", "", base).strip()
    if no_paren and no_paren != base:
        variants.append(no_paren)
    return variants


def _name_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^\w']+", fold(strip_possessives(text)), flags=re.UNICODE)
            if t and t not in {"the", "a", "an"}]


def _is_subsequence(short: List[str], long: List[str]) -> bool:
    it = iter(long)
    return all(tok in it for tok in short)


def match_question_entities(
    question: str,
    entity_names: Iterable[str],
    stopwords: Sequence[str] = (),
    max_per_span: int = 3,
) -> List[str]:
    """Selects graph entities mentioned in a question.

    Matching tiers (all case-insensitive, possessives normalised on both sides):
      1. exact word-boundary occurrence of an entity name in the question;
      2. a capitalised question span that is a word-boundary prefix of an entity name
         ("Caroline LeRoy" -> "Caroline LeRoy Webster", "The Unwinding" -> "Unwinding: An ...").
    Overlapping matches are resolved greedily: capitalised > longer > exact. Matches written
    entirely in lower case in the question (generic nouns such as "employer", "author") are
    only used when no capitalised match exists.
    """
    q = strip_possessives(question)
    stop = {s.lower() for s in stopwords}
    names = list(dict.fromkeys(n for n in entity_names if n))
    spans = capitalised_spans(q)
    # (start, end, is_capitalised, is_exact, name)
    candidates: List[Tuple[int, int, bool, bool, str]] = []

    for name in names:
        for variant in _name_variants(name):
            if len(variant) < 2 or variant.lower() in stop or variant.lower() in _FUNCTION_WORDS:
                continue
            pattern = r"(?<!\w)" + re.escape(variant) + r"(?!\w)"
            for m in re.finditer(pattern, q, re.IGNORECASE):
                is_cap = any(_is_capitalised(t) for t in m.group(0).split())
                candidates.append((m.start(), m.end(), is_cap, True, name))

        name_tokens = _name_tokens(name)
        for start, end, text in spans:
            span_tokens = _name_tokens(text)
            if (len(span_tokens) >= 2 and len(name_tokens) > len(span_tokens)
                    and len(name_tokens) <= len(span_tokens) + 2
                    and name_tokens[0] == span_tokens[0] and name_tokens[-1] == span_tokens[-1]
                    and _is_subsequence(span_tokens, name_tokens)):
                # "Frances Tupper" -> "Frances Amélia Tupper"; "Philippe, Duke of Orléans" ->
                # "Philippe I, Duke of Orléans"
                candidates.append((start, end, True, False, name))

        folded_name = fold(strip_possessives(name))
        for start, end, text in spans:
            options = [(start, text)]
            art = _LEADING_ARTICLE_RE.match(text)
            if art:
                options.append((start + art.end(), text[art.end():]))
            for s_start, s_text in options:
                folded_span = fold(s_text)
                if not folded_span or folded_span == folded_name or not folded_name.startswith(folded_span):
                    continue
                rest = folded_name[len(folded_span):]
                n_tokens = len(s_text.split())
                if rest[:1] in (":", "(") or rest[:2] in (" (", " -", ": ") or (rest[:1] == " " and n_tokens >= 2):
                    candidates.append((s_start, end, True, False, name))

    if not candidates:
        return []
    if any(c[2] for c in candidates):
        candidates = [c for c in candidates if c[2]]

    candidates.sort(key=lambda c: (-(c[1] - c[0]), not c[3], c[0]))
    chosen_spans: List[Tuple[int, int]] = []
    selected: List[str] = []
    per_span: dict = {}
    for start, end, _cap, exact, name in candidates:
        overlaps = [s for s in chosen_spans if not (end <= s[0] or start >= s[1])]
        if overlaps and (start, end) not in chosen_spans:
            continue
        key = (start, end)
        bucket = per_span.setdefault(key, {"exact": exact, "names": []})
        if bucket["exact"] and not exact:
            continue
        if len(bucket["names"]) >= max_per_span or name in bucket["names"]:
            continue
        bucket["names"].append(name)
        if key not in chosen_spans:
            chosen_spans.append(key)
        selected.append(name)

    ordered = sorted(per_span.items(), key=lambda kv: kv[0][0])
    result: List[str] = []
    for _, bucket in ordered:
        for n in bucket["names"]:
            if n in selected and n not in result:
                result.append(n)
    return result


# --------------------------------------------------------------------------------------
# Placeholders, possessive associations, acronym aliases, relevance stems
# --------------------------------------------------------------------------------------

_PLACEHOLDERS = {
    "unknown", "unspecified", "unnamed", "unidentified", "n/a", "na", "none", "null", "nil",
    "someone", "somebody", "something", "anyone", "anything", "various", "others", "other",
    "it", "he", "she", "they", "them", "him", "her", "his", "its", "their", "we", "us", "you",
    "this", "that", "these", "those", "people", "person", "individual", "entity", "thing",
}


def is_placeholder(value: str) -> bool:
    """Extractor filler values that do not denote a specific entity."""
    v = _LEADING_ARTICLE_RE.sub("", (value or "").strip().strip(".,;:\"'")).strip().casefold()
    return not v or v in _PLACEHOLDERS


_NAME_TOKEN = r"[A-Z0-9][\w&.\-]*"
_POSSESSIVE_ASSOC_RE = re.compile(
    r"^(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r")*)['’]s\s+(" + _NAME_TOKEN + r"(?:\s+(?:of|for|and|the|" + _NAME_TOKEN + r"))*)$"
)


def parse_possessive_association(value: str) -> Optional[Tuple[str, str]]:
    """"DLR's Lander Control Center" -> ("DLR", "Lander Control Center").

    Only fires when both the possessor and the possessed are written as proper names, so
    descriptive phrases ("Swift's frustrations", "Belle's brothers") are left alone.
    """
    m = _POSSESSIVE_ASSOC_RE.match((value or "").strip())
    return (m.group(1), m.group(2)) if m else None


_ACRONYM_PAREN_RE = re.compile(r"\(\s*([A-Z][A-Z0-9&.\-]{1,9})\s*\)")
_ALIAS_CONNECTORS = {"of", "for", "and", "the", "de", "du", "des", "für", "&"}


def _initials_contain(acronym: str, words: List[str]) -> bool:
    letters = [c for c in acronym.upper() if c.isalpha()]
    initials = "".join(w[0].upper() for w in words if w)
    it = iter(initials)
    return all(ch in it for ch in letters)


def _name_before(text: str) -> List[str]:
    """Maximal run of capitalised tokens (with name connectors) ending right before a position."""
    tokens = re.findall(r"[^\s,;:()]+", text)
    name_tokens: List[str] = []
    for tok in reversed(tokens):
        if name_tokens and tok.endswith("."):
            break  # sentence boundary: "...Cooperative. The Southern ..."
        if re.search(r"\d", tok):
            break  # tables / numbers are never part of a name
        if tok[:1].isupper():
            name_tokens.insert(0, tok)
        elif tok.lower() in _ALIAS_CONNECTORS and name_tokens:
            name_tokens.insert(0, tok)
        else:
            break
    while name_tokens and name_tokens[0].lower() in _ALIAS_CONNECTORS:
        name_tokens.pop(0)
    return name_tokens


def _shortest_aligned_suffix(name_tokens: List[str], acronym: str) -> Optional[List[str]]:
    """Shortest suffix that starts on the abbreviation's first letter and whose capitalised
    initials still contain the abbreviation as a subsequence."""
    for i in range(len(name_tokens) - 1, -1, -1):
        suffix = name_tokens[i:]
        if suffix[0][:1].upper() != acronym[:1].upper():
            continue
        if _initials_contain(acronym, [t for t in suffix if t[:1].isupper()]):
            return suffix
    return None


def mine_acronym_aliases(text: str) -> List[Tuple[str, str]]:
    """Finds "Long Name (ABBR)" definitions in source text.

    The abbreviation must be alphabetic (no digits) and spell the capitalised initials of the
    preceding name; the long name is trimmed to the shortest aligned suffix ("... North Carolina
    on the National Register of Historic Places (NRHP)" -> "National Register of Historic
    Places"). Names never extend across a sentence boundary or through numbers. Abbreviations
    that do not align with the initials (other languages, "(USA)", "(HD)") are not mined:
    a wrong alias silently merges unrelated entities, a missing one only loses a link.
    """
    aliases: List[Tuple[str, str]] = []
    for m in _ACRONYM_PAREN_RE.finditer(text or ""):
        acronym = m.group(1).strip(".")
        if re.search(r"\d", acronym) or len([c for c in acronym if c.isalpha()]) < 2:
            continue
        name_tokens = _name_before(text[: m.start()].rstrip())
        if not name_tokens:
            continue
        aligned = _shortest_aligned_suffix(name_tokens, acronym)
        if not aligned:
            continue
        long_name = " ".join(aligned).rstrip(".")
        if fold(long_name) != fold(acronym):
            aliases.append((long_name, acronym))
    return list(dict.fromkeys(aliases))


_STEM_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ed", "es", "s")
_RELEVANCE_STOP = {
    "what", "which", "where", "when", "who", "whom", "whose", "how", "why", "the", "a", "an", "of",
    "is", "was", "are", "were", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or",
    "that", "this", "does", "did", "do", "thing", "person", "entity",
}


def stem(word: str) -> str:
    w = word.lower()
    for suf in _STEM_SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> set:
    """Crude content-word stems used to rank evidence by relevance to the question."""
    return {stem(w) for w in re.findall(r"[A-Za-z]+", text or "") if w.lower() not in _RELEVANCE_STOP and len(w) > 2}