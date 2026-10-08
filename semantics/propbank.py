"""PropBank loader that parses official PropBank 3.4 XML frame files."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

from semantics.frames import all_system_frames

_CORE_ROLE_RE = re.compile(r"^:ARG([0-5])$")


def _canonical_roleset_id(raw: str) -> str:
    """'abandon.01' / 'abandon-01' / 'add_up.01' -> canonical '<lemma>.<NN>'.

    Only the trailing sense separator is normalised; hyphens/underscores inside the lemma
    (e.g. 'co-write', 'add_up') are preserved so multi-word lemmas keep their identity.
    """
    raw = (raw or "").strip().lower()
    m = re.match(r"^(.*?)[.\-_](\d{1,2})$", raw)
    if not m:
        return raw
    return f"{m.group(1)}.{int(m.group(2)):02d}"


def _lemma_of(roleset_id: str) -> str:
    return roleset_id.rsplit(".", 1)[0]


class PropBankRoleset(BaseModel):
    roleset_id: str
    name: str
    description: str
    roles: Dict[str, str]  # e.g., {":ARG0": "employer", ":ARG1": "employee"}


class PropBankCatalog:
    """Manages indexed access to downloaded official PropBank XML frame files."""

    DEFAULT_FRAMES_DIR = Path("data/propbank-frames/frames")
    CACHE_PATH = Path("data/propbank_cache.json")

    STANDARD_MODIFIERS = {
        ":manner", ":purpose", ":direction", ":location", ":time",
        ":argm-loc", ":argm-tmp", ":argm-mnr", ":argm-adv", ":argm-cau",
        ":argm-dir", ":argm-dis", ":argm-ext", ":argm-gol", ":argm-prd",
        ":argm-prp", ":argm-neg", ":argm-mod", ":argm-adj", ":argm-com",
    }

    def __init__(
        self,
        frames_dir: Optional[Union[Path, str]] = None,
        cache_file: Optional[Union[Path, str]] = None,
    ):
        self.frames_dir = Path(frames_dir) if frames_dir else self.DEFAULT_FRAMES_DIR
        self.cache_file = Path(cache_file) if cache_file else self.CACHE_PATH
        # The lock written by download_propbank.py identifies the exact frames release in use.
        lock_file = self.cache_file.parent / "propbank_lock.json"
        try:
            self.lock: Optional[Dict] = json.loads(lock_file.read_text(encoding="utf-8")) if lock_file.exists() else None
        except (OSError, json.JSONDecodeError):
            self.lock = None
        self.framesets: Dict[str, PropBankRoleset] = {}
        self.lemma_to_senses: Dict[str, List[str]] = {}
        self._load()
        self._apply_canonical_frames()

    def _apply_canonical_frames(self) -> None:
        """System-defined frames take precedence so validation matches the extraction contract."""
        for rid, (label, roles) in all_system_frames().items():
            self.framesets[rid] = PropBankRoleset(roleset_id=rid, name=label.lower(), description=label.lower(), roles=dict(roles))
        self._index()

    def _index(self) -> None:
        self.lemma_to_senses.clear()
        for k in self.framesets:
            self.lemma_to_senses.setdefault(_lemma_of(k), []).append(k)

    @property
    def commit(self) -> Optional[str]:
        return (self.lock or {}).get("commit")

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # The cache is valid only for the frames release it was built from.
                cached_commit = (data.get("_meta") or {}).get("commit") if "framesets" in data else None
                if cached_commit != self.commit:
                    raise ValueError("PropBank cache built from a different frames release")
                for k, v in data.get("framesets", data).items():
                    rid = _canonical_roleset_id(k)
                    v = dict(v)
                    v["roleset_id"] = rid
                    self.framesets[rid] = PropBankRoleset(**v)
                self._index()
                return
            except Exception:
                self.framesets.clear()

        if self.frames_dir.exists() and any(self.frames_dir.glob("*.xml")):
            self.parse_xml_directory(self.frames_dir)
        else:
            print(
                f"Notice: Official PropBank frames not found at {self.frames_dir}. "
                "Run `python download_propbank.py` to acquire official XML framesets."
            )
        if self.framesets and not self.lock:
            print("Notice: PropBank frames are not pinned (no data/propbank_lock.json). "
                  "Run `python download_propbank.py` before experiments you intend to report.")

    def parse_xml_directory(self, frames_dir: Union[Path, str]) -> int:
        frames_path = Path(frames_dir)
        compiled: Dict[str, Dict] = {}
        for xml_file in frames_path.glob("*.xml"):
            try:
                root = ET.parse(xml_file).getroot()
            except ET.ParseError:
                continue
            for pred in root.findall("predicate"):
                for roleset in pred.findall("roleset"):
                    rid = _canonical_roleset_id(roleset.get("id", ""))
                    if not rid or "." not in rid:
                        continue
                    name = roleset.get("name", "").strip()
                    roles = {}
                    roles_elem = roleset.find("roles")
                    if roles_elem is not None:
                        for r in roles_elem.findall("role"):
                            n = r.get("n", "")
                            if n.isdigit():
                                roles[f":ARG{n}"] = r.get("descr", "").strip()
                    compiled[rid] = {"roleset_id": rid, "name": name, "description": name, "roles": roles}

        if compiled:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({"_meta": {"commit": self.commit, "ref": (self.lock or {}).get("ref")},
                           "framesets": compiled}, f, indent=2)
            self.framesets = {k: PropBankRoleset(**v) for k, v in compiled.items()}
            self._apply_canonical_frames()
        return len(self.framesets)

    def get_candidate_senses(self, lemma: str) -> List[PropBankRoleset]:
        """Returns all rolesets for a lemma, trying '_', '-' and space spellings."""
        clean = lemma.strip().lower()
        for variant in dict.fromkeys([clean, clean.replace("_", "-"), clean.replace("-", "_"), clean.replace(" ", "_"), clean.replace(" ", "-")]):
            sense_ids = self.lemma_to_senses.get(variant)
            if sense_ids:
                return [self.framesets[sid] for sid in sense_ids if sid in self.framesets]
        return []

    # Added by the normalization layer to link the owner named in a possessive argument.
    POSSESSOR_ROLE = ":ARGM-POSS"

    def role_description(self, sense_id: str, role: str) -> str:
        if role == self.POSSESSOR_ROLE:
            return "possessor"
        roleset = self.framesets.get(sense_id)
        return roleset.roles.get(role, "") if roleset else ""

    def disambiguate_sense(self, lemma: str, sentence_context: str) -> Optional[PropBankRoleset]:
        """Picks the sense whose name/role glosses overlap most with the context (bag of words)."""
        candidates = self.get_candidate_senses(lemma)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        words = set(re.findall(r"\w+", sentence_context.lower()))
        best_candidate = candidates[0]
        max_overlap = -1
        for c in candidates:
            sense_words = set(re.findall(r"\w+", (c.name + " " + c.description + " " + " ".join(c.roles.values())).lower()))
            overlap = len(words & sense_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_candidate = c
        return best_candidate

    def _is_modifier(self, role: str) -> bool:
        r_lower = role.lower()
        return r_lower in self.STANDARD_MODIFIERS or r_lower.startswith(":argm-")

    def validate_roles(self, sense_id: str, proposed_roles: Dict[str, str]) -> Dict[str, str]:
        """Aligns proposed arguments with the roleset signature without disconnecting entities.

        * Roleset without :ARG0 but :ARG0 proposed (unaccusative / copular frames): the proposed
          core arguments are shifted onto the declared slots in order (order is informative here).
        * Otherwise declared arguments stay; undeclared ones move into free declared slots other
          than :ARG0. :ARG0 denotes the agent, which an argument's position cannot establish
          (a passive "owned by X" and a location "in X" have the same ARG1+ARG2 shape).
        * Arguments that still do not fit are kept under their own label if they are one of the
          positional slots the extraction schema exposes (:ARG0-:ARG2), so the entity stays
          linked; higher undeclared arguments (:ARG3-:ARG5) are treated as phantom and dropped.
        Modifiers and contextual roles are always kept.
        """
        modifiers = {r: v for r, v in proposed_roles.items() if self._is_modifier(r)}
        core = {r: v for r, v in proposed_roles.items() if _CORE_ROLE_RE.match(r)}

        roleset = self.framesets.get(sense_id)
        if not roleset or not roleset.roles:
            return {**core, **modifiers}

        declared = sorted((r for r in roleset.roles if _CORE_ROLE_RE.match(r)), key=lambda r: int(r[-1]))
        proposed = sorted(core, key=lambda r: int(r[-1]))
        result: Dict[str, str] = {}

        if ":ARG0" in core and ":ARG0" not in declared and declared:
            for src, dst in zip(proposed, declared):
                result[dst] = core[src]
        else:
            for r in proposed:
                if r in declared:
                    result[r] = core[r]
            free = [d for d in declared if d not in result and d != ":ARG0"]
            for src in (r for r in proposed if r not in declared):
                if free:
                    result[free.pop(0)] = core[src]
                elif int(src[-1]) <= 2 and src not in result:
                    result[src] = core[src]

        result.update(modifiers)
        return result

GLOBAL_CATALOG = PropBankCatalog()