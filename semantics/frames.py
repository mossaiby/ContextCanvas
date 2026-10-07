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


def all_system_frames() -> Dict[str, Tuple[str, Dict[str, str]]]:
    return {**CANONICAL_FRAMES, **INTERNAL_FRAMES}


def prompt_frame_lines() -> List[str]:
    lines = []
    for rid, (label, roles) in CANONICAL_FRAMES.items():
        role_txt = ", ".join(f"'{r}': {d}" for r, d in roles.items())
        lines.append(f"   - {label}: {rid} {{{role_txt}}}")
    return lines
