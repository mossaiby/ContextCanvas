"""Canonical relation frames defined by Context Canvas itself.

These frames are the contract between the extraction prompt, the pipeline repairs (which
rewrite kinship/birth/location facts into them) and role validation. They are merged into the
PropBank catalog with precedence, so validation can never disagree with what the prompt asked
the extractor to produce, regardless of how a given PropBank release defines the same id.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# roleset id -> (prompt label, {role: description})
CANONICAL_FRAMES: Dict[str, Tuple[str, Dict[str, str]]] = {
    "own.01": ("OWNERSHIP", {":ARG0": "owner", ":ARG1": "possession"}),
    "locate.01": ("LOCATION", {":ARG1": "thing located", ":ARG2": "location"}),
    "marry.01": ("SPOUSE", {":ARG0": "spouse", ":ARG1": "other spouse"}),
    "parent.01": ("PARENT/CHILD (one event per parent)", {":ARG0": "parent", ":ARG1": "child"}),
    "sibling.01": ("SIBLING", {":ARG0": "person", ":ARG1": "sibling"}),
    "bear.02": ("BIRTH (birthplace in spatial_context, date in time_context)", {":ARG0": "parent giving birth", ":ARG1": "person born"}),
    "employ.01": ("EMPLOYMENT", {":ARG0": "employer", ":ARG1": "employee", ":ARG2": "position"}),
    "member.01": ("MEMBERSHIP", {":ARG0": "member", ":ARG1": "group"}),
    "found.01": ("FOUNDING", {":ARG0": "founder", ":ARG1": "organisation founded"}),
    "distribute.01": ("DISTRIBUTION", {":ARG0": "distributor", ":ARG1": "thing distributed", ":ARG2": "recipient"}),
    "manufacture.01": ("MANUFACTURE", {":ARG0": "manufacturer", ":ARG1": "product"}),
    "partner.01": ("BUSINESS/CREATIVE PARTNERSHIP (never marriage)", {":ARG0": "partner", ":ARG1": "other partner"}),
}

# Frames produced by the pipeline itself (never requested from the extractor).
INTERNAL_FRAMES: Dict[str, Tuple[str, Dict[str, str]]] = {
    "alias.01": ("ALIAS", {":ARG0": "entity", ":ARG1": "alternative name of the same entity"}),
    "affiliate.01": ("POSSESSIVE ASSOCIATION", {":ARG0": "possessor / parent organisation", ":ARG1": "associated entity"}),
}

# Relations whose two arguments denote the same real-world entity; traversal crosses them freely.
IDENTITY_FRAMES = {"alias.01"}

# The verbs allowed to use each system frame. An extractor attaches system frames to verbs that do
# not express them ("hold" -> locate.01, "be home to" -> member.01); such a fact would carry a
# false meaning (containment is transitive, so one wrong locate.01 corrupts every place chain
# through it). Any other verb keeps its own sense: a fact without role meanings is harmless, a
# fact with wrong ones is not. Lemmas are compared after lemma_key().
FRAME_VERBS: Dict[str, set] = {
    "locate.01": {"locate", "be located", "situate", "be situated", "lie", "sit", "stand", "headquarter",
                  "headquarters", "be headquartered", "take place"},
    "own.01": {"own", "possess", "acquire", "buy", "purchase"},
    "marry.01": {"marry", "wed", "be married"},
    "parent.01": {"parent", "father", "mother", "beget"},
    "sibling.01": {"sibling", "brother", "sister"},
    "bear.02": {"bear", "born", "be born", "give birth"},
    "employ.01": {"employ", "hire", "work for", "work at"},
    "member.01": {"member", "be member", "be a member", "join", "belong", "belong to", "play for"},
    "found.01": {"found", "co-found", "cofound", "establish", "co-establish", "set up"},
    "distribute.01": {"distribute", "release", "market", "publish"},
    "manufacture.01": {"manufacture", "make", "build"},
    "partner.01": {"partner", "collaborate", "team up"},
    "alias.01": {"alias"},
    "affiliate.01": {"affiliate"},
}


# The other direction of the contract: a listed verb whose extracted sense does not exist
# ("co-found" -> "co-found.01") takes its system frame. Internal frames are produced only by
# the normalization layer, so they are not reachable from verbs.
VERB_FRAMES: Dict[str, str] = {
    verb: sense for sense, verbs in FRAME_VERBS.items() if sense in CANONICAL_FRAMES for verb in verbs
}


def lemma_key(lemma: str) -> str:
    """Comparable form of a lemma: lower case, underscores as spaces ("be_located" == "Be located")."""
    return " ".join((lemma or "").lower().replace("_", " ").split())


def frame_accepts(sense_id: str, lemma: str) -> bool:
    """Whether a verb may carry the given sense. Only system frames are restricted."""
    verbs = FRAME_VERBS.get(sense_id)
    return verbs is None or lemma_key(lemma) in verbs


def all_system_frames() -> Dict[str, Tuple[str, Dict[str, str]]]:
    return {**CANONICAL_FRAMES, **INTERNAL_FRAMES}


def prompt_frame_lines() -> List[str]:
    lines = []
    for rid, (label, roles) in CANONICAL_FRAMES.items():
        role_txt = ", ".join(f"'{r}': {d}" for r, d in roles.items())
        lines.append(f"   - {label}: {rid} {{{role_txt}}}")
    return lines