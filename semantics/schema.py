"""Strict Pydantic v2 schemas for semantic extraction, context modeling, and adaptive role canonicalization."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from semantics.text_utils import split_place

ROLE_ALIAS_MAP: Dict[str, str] = {
    # Agentive / Initiator / Subject roles -> :ARG0
    ":actor": ":ARG0", ":agent": ":ARG0", ":subject": ":ARG0", ":creator": ":ARG0", ":create": ":ARG0",
    ":author": ":ARG0", ":producer": ":ARG0", ":produce": ":ARG0", ":director": ":ARG0", ":direct": ":ARG0",
    ":founder": ":ARG0", ":found": ":ARG0", ":funder": ":ARG0", ":fund": ":ARG0", ":employer": ":ARG0",
    ":speaker": ":ARG0", ":caller": ":ARG0", ":writer": ":ARG0", ":write": ":ARG0", ":inventor": ":ARG0",
    ":invent": ":ARG0", ":parent": ":ARG0", ":performer": ":ARG0", ":artist": ":ARG0", ":musician": ":ARG0",
    ":singer": ":ARG0", ":player": ":ARG0", ":member": ":ARG0", ":leader": ":ARG0", ":originator": ":ARG0",
    ":distributor": ":ARG0", ":publisher": ":ARG0", ":manufacturer": ":ARG0", ":maker": ":ARG0",
    ":owner": ":ARG0", ":proprietor": ":ARG0", ":operator": ":ARG0", ":builder": ":ARG0", ":supplier": ":ARG0",
    ":seller": ":ARG0", ":buyer": ":ARG0", ":parent_company": ":ARG0", ":holding_company": ":ARG0",
    ":founded_by": ":ARG0", ":distributed_by": ":ARG0", ":manufactured_by": ":ARG0", ":owned_by": ":ARG0",
    ":headquartered_by": ":ARG0", ":partner_a": ":ARG0",

    # Patient / Theme / Target roles -> :ARG1
    ":patient": ":ARG1", ":theme": ":ARG1", ":object": ":ARG1", ":entity": ":ARG1", ":creation": ":ARG1",
    ":product": ":ARG1", ":work": ":ARG1", ":novel": ":ARG1", ":play": ":ARG1", ":target": ":ARG1",
    ":problem": ":ARG1", ":callee": ":ARG1", ":recipient": ":ARG1", ":topic": ":ARG1", ":child": ":ARG1",
    ":sibling": ":ARG1", ":spouse": ":ARG1", ":partner": ":ARG1", ":partner_b": ":ARG1", ":wife": ":ARG1",
    ":husband": ":ARG1", ":album": ":ARG1", ":song": ":ARG1", ":book": ":ARG1", ":movie": ":ARG1",
    ":film": ":ARG1", ":organization": ":ARG1", ":group": ":ARG1", ":band": ":ARG1", ":company": ":ARG1",
    ":university": ":ARG1", ":institution": ":ARG1", ":subsidiary": ":ARG1", ":division": ":ARG1",
    ":facility": ":ARG1", ":property": ":ARG1", ":building": ":ARG1", ":aircraft": ":ARG1",
    ":vehicle": ":ARG1", ":distributee": ":ARG1", ":manufacturee": ":ARG1", ":release": ":ARG1",

    # Secondary arguments / Attributes -> :ARG2
    ":example": ":ARG2", ":description": ":ARG2", ":instrument": ":ARG2", ":attribute": ":ARG2",
    ":capital": ":ARG2", ":name": ":ARG2", ":beneficiary": ":ARG2", ":role": ":ARG2", ":title": ":ARG2",
    ":status": ":ARG2", ":genre": ":ARG2", ":headquarters": ":ARG2", ":headquarter": ":ARG2", ":hq": ":ARG2",
    ":manufacturer_of": ":ARG2", ":distributor_of": ":ARG2",

    # Locative descriptors -> :location
    ":place": ":location", ":region": ":location", ":city": ":location", ":venue": ":location",
    ":site": ":location", ":where": ":location", ":country": ":location", ":state": ":location",
    ":birthplace": ":location", ":birth_place": ":location", ":headquartered_in": ":location",
    ":located_in": ":location", ":based_in": ":location", ":residence": ":location",
    ":territory": ":location", ":municipality": ":location", ":province": ":location",

    # Temporal descriptors -> :time
    ":date": ":time", ":year": ":time", ":period": ":time", ":when": ":time", ":timestamp": ":time",
    ":birth_date": ":time", ":start_date": ":time", ":end_date": ":time", ":release_date": ":time",
    ":release_year": ":time", ":founded_year": ":time", ":founded_date": ":time",
}

CONTEXT_ROLES = {":manner", ":purpose", ":direction", ":location", ":time"}


class TemporalContext(BaseModel):
    """Temporal scope supporting points and intervals with normalized timestamps."""
    raw_expression: str = Field(default="", description="The exact text string indicating time.")
    start_year: Optional[int] = Field(default=None, description="Start year parsed from expression.")
    end_year: Optional[int] = Field(default=None, description="End year parsed from expression.")

    @model_validator(mode="before")
    @classmethod
    def coerce_temporal(cls, v: Any) -> Any:
        if v is None:
            return {"raw_expression": ""}
        if isinstance(v, str):
            return {"raw_expression": v.strip()}
        if isinstance(v, (int, float)):
            return {"raw_expression": str(v), "start_year": int(v), "end_year": int(v)}
        if isinstance(v, dict):
            raw = v.get("raw_expression") or v.get("expression") or v.get("text") or v.get("time") or ""
            s_yr, e_yr = v.get("start_year"), v.get("end_year")
            try:
                s_yr = int(s_yr) if s_yr is not None else None
            except (ValueError, TypeError):
                s_yr = None
            try:
                e_yr = int(e_yr) if e_yr is not None else None
            except (ValueError, TypeError):
                e_yr = None
            return {
                "raw_expression": str(raw).strip() if raw is not None else "",
                "start_year": s_yr,
                "end_year": e_yr,
            }
        return {"raw_expression": ""}

    @model_validator(mode="after")
    def populate_years(self) -> "TemporalContext":
        if self.start_year is None and self.raw_expression:
            matches = re.findall(r"\b(1\d{3}|20\d{2})\b", self.raw_expression)
            if len(matches) == 1:
                self.start_year = int(matches[0])
                self.end_year = int(matches[0])
            elif len(matches) >= 2:
                self.start_year = int(matches[0])
                self.end_year = int(matches[1])
        if self.start_year is not None and self.end_year is None:
            self.end_year = self.start_year
        return self


class SpatialContext(BaseModel):
    """Spatial scope modeling locations and parent regions."""
    location_name: str = Field(default="", description="The primary geographical entity or site.")
    parent_region: Optional[str] = Field(default=None, description="Enclosing administrative boundary.")

    @model_validator(mode="before")
    @classmethod
    def coerce_spatial(cls, v: Any) -> Any:
        if v is None:
            return {"location_name": ""}
        if isinstance(v, str):
            return {"location_name": v.strip()}
        if isinstance(v, dict):
            loc = v.get("location_name") or v.get("location") or v.get("name") or v.get("place") or ""
            parent = v.get("parent_region")
            return {
                "location_name": str(loc).strip() if loc is not None else "",
                "parent_region": str(parent).strip() if parent else None,
            }
        return {"location_name": ""}

    @model_validator(mode="after")
    def split_qualified_location(self) -> "SpatialContext":
        """"Southwest City, Missouri" -> location 'Southwest City', parent 'Missouri'.

        Without this the qualified string becomes its own graph entity and never joins the
        node created for the bare place name elsewhere in the corpus.
        """
        parts = split_place(self.location_name)
        if parts:
            self.location_name = parts[0]
            tail = ", ".join(parts[1:])
            self.parent_region = f"{tail}, {self.parent_region}" if self.parent_region else tail
        return self


class EpistemicContext(BaseModel):
    """Provenance and probabilistic certainty framing."""
    source: str = Field(default="Unattributed", description="Originating document, report, or speaker.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    is_speculative: bool = Field(default=False, description="True if based on unverified rumor or leak.")


class ExtractedEvent(BaseModel):
    """An asserted event frame with assigned roles and surrounding context envelope."""
    temp_id: str = Field(default="ev_1", description="Unique reference ID for this clause.")
    lemma: str = Field(default="event", description="Base dictionary verb or nominal form.")
    sense_id: str = Field(default="event.01", description="PropBank roleset ID matching defined frame.")
    roles: Dict[str, str] = Field(default_factory=dict, description="Mapping of PropBank argument tags to entity names.")
    time_context: Optional[TemporalContext] = None
    spatial_context: Optional[SpatialContext] = None
    epistemic_context: EpistemicContext = Field(default_factory=EpistemicContext)

    @model_validator(mode="before")
    @classmethod
    def pre_validate_event(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "temp_id" not in data or not data["temp_id"]:
                data["temp_id"] = data.get("id") or "ev_1"
            # Extractors sometimes write the sense into the lemma field ("lemma": "bear.02").
            # Left alone, that yields the lemma "bear.02" and the nonsense sense "bear_02.01", which
            # matches no frame, so the event gets no role meanings and escapes every repair.
            embedded = re.fullmatch(r"\s*([A-Za-z][A-Za-z_\- ]*?)[._-](\d{1,2})\s*", str(data.get("lemma") or ""))
            if embedded:
                base, number = embedded.group(1).strip(), int(embedded.group(2))
                data["lemma"] = base
                sense = str(data.get("sense_id") or "").strip().lower()
                if not re.fullmatch(re.escape(base.lower()) + r"\.\d{2}", sense):
                    data["sense_id"] = f"{base}.{number:02d}"
            if "sense_id" not in data or not data["sense_id"]:
                alt_sense = data.get("frame") or data.get("roleset_id") or data.get("roleset") or data.get("sense")
                if alt_sense:
                    data["sense_id"] = str(alt_sense)
                elif data.get("lemma"):
                    data["sense_id"] = f"{data['lemma']}.01"
                else:
                    data["sense_id"] = "event.01"
            if "lemma" not in data or not data["lemma"]:
                if "sense_id" in data and "." in str(data["sense_id"]):
                    data["lemma"] = str(data["sense_id"]).rsplit(".", 1)[0]
                elif "frame" in data and ("." in str(data["frame"]) or "-" in str(data["frame"])):
                    data["lemma"] = re.split(r"[.\-]\d", str(data["frame"]))[0]
                else:
                    data["lemma"] = "event"
        return data

    @field_validator("sense_id", mode="before")
    @classmethod
    def validate_sense_format(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return "event.01"
        clean = v.strip().lower()
        m_num = re.search(r"[.\-_](\d{1,2})$", clean)
        num_str = f"{int(m_num.group(1)):02d}" if m_num else "01"
        lemma_part = re.sub(r"[.\-_]\d{1,2}$", "", clean).strip()
        lemma_clean = re.sub(r"[^\w\-]", "_", lemma_part).strip("_-")
        if not lemma_clean:
            lemma_clean = "event"
        return f"{lemma_clean}.{num_str}"

    @field_validator("roles", mode="before")
    @classmethod
    def coerce_and_validate_roles(cls, v: Any) -> Dict[str, str]:
        raw_dict: Dict[str, Any] = {}
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    role_key = item.get("role") or item.get("argument") or item.get("name") or item.get("tag") or item.get("label")
                    val = item.get("value") or item.get("entity") or item.get("text") or item.get("val") or item.get("target")
                    if role_key and val:
                        raw_dict[str(role_key)] = str(val)
                    elif len(item) == 1:
                        k, val = next(iter(item.items()))
                        raw_dict[str(k)] = str(val)
        elif isinstance(v, dict):
            raw_dict = v
        else:
            return {}

        normalized: Dict[str, str] = {}
        for k, val in raw_dict.items():
            if not k or not str(k).strip() or not val or not str(val).strip():
                continue
            cleaned_val = str(val).strip()
            key = str(k).strip()
            if key in {":", ""}:
                continue
            if not key.startswith(":"):
                key = f":{key}"

            arg_match = re.match(r"^:arg([0-5])$", key, re.IGNORECASE)
            if arg_match:
                normalized[f":ARG{arg_match.group(1)}"] = cleaned_val
                continue
            argm_match = re.match(r"^:argm-([a-z]+)$", key, re.IGNORECASE)
            if argm_match:
                normalized[f":ARGM-{argm_match.group(1).upper()}"] = cleaned_val
                continue
            lowered_key = key.lower()
            if lowered_key in ROLE_ALIAS_MAP:
                target = ROLE_ALIAS_MAP[lowered_key]
                if target not in normalized:
                    normalized[target] = cleaned_val
                continue
            if lowered_key in CONTEXT_ROLES:
                normalized[lowered_key] = cleaned_val
                continue
            raise ValueError(f"Invalid PropBank role identifier: {k}")

        return normalized


class DiscourseRelation(BaseModel):
    """Document-level inter-event relationship."""
    source_id: str = Field(default="", description="Source event ID.")
    target_id: str = Field(default="", description="Target event ID.")
    relation: str = Field(default=":cause", pattern=r"^:(cause|result|before|after|condition|contrast)$")

    @model_validator(mode="before")
    @classmethod
    def coerce_discourse(cls, v: Any) -> Any:
        if isinstance(v, dict):
            if "source_id" not in v:
                v["source_id"] = v.get("source") or v.get("src") or v.get("from") or ""
            if "target_id" not in v:
                v["target_id"] = v.get("target") or v.get("tgt") or v.get("to") or ""
            if "relation" not in v:
                v["relation"] = v.get("rel") or v.get("type") or ":cause"
        return v

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return ":cause"
        clean = v.strip().lower()
        if not clean.startswith(":"):
            clean = f":{clean}"
        valid = {":cause", ":result", ":before", ":after", ":condition", ":contrast"}
        if clean in valid:
            return clean
        if "before" in clean or "prior" in clean:
            return ":before"
        if "after" in clean:
            return ":after"
        return ":cause"


def _heuristic_role_for_key(k_lower: str) -> str:
    """Maps an unanticipated role key onto a canonical slot by keyword."""
    tokens = set(re.split(r"[^a-z]+", k_lower))
    if tokens & {"by", "agent", "actor", "from", "source", "author", "creator", "founder", "owner", "employer", "performer", "subject"}:
        return ":ARG0"
    if tokens & {"to", "target", "theme", "object", "patient", "product", "film", "work", "song", "album"}:
        return ":ARG1"
    if tokens & {"loc", "location", "place", "city", "country", "state", "where", "site", "headquarter", "headquarters"}:
        return ":location"
    if tokens & {"time", "date", "year", "when"}:
        return ":time"
    return ":ARG2"


class ExtractionPayload(BaseModel):
    """Root JSON structure emitted by extraction models."""
    events: List[ExtractedEvent] = Field(default_factory=list)
    discourse: List[DiscourseRelation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_payload(cls, v: Any) -> Any:
        if isinstance(v, list):
            v = {"events": v, "discourse": []}
        elif isinstance(v, dict) and "events" not in v:
            if "roles" in v or "lemma" in v or "sense_id" in v or "frame" in v:
                v = {"events": [v], "discourse": []}

        if isinstance(v, dict) and "events" in v and isinstance(v["events"], list):
            v["events"] = [ev for ev in v["events"] if isinstance(ev, dict)]
            for ev in v["events"]:
                if "roles" in ev and isinstance(ev["roles"], dict):
                    clean_roles: Dict[str, str] = {}
                    deferred: List[tuple] = []
                    for r_k, r_v in ev["roles"].items():
                        if not r_v or isinstance(r_v, (dict, list)):
                            continue
                        k_clean = str(r_k).strip()
                        if not k_clean.startswith(":"):
                            k_clean = f":{k_clean}"
                        k_lower = k_clean.lower()
                        if (
                            re.match(r"^:arg[0-5]$", k_lower)
                            or re.match(r"^:argm-[a-z]+$", k_lower)
                            or k_lower in ROLE_ALIAS_MAP
                            or k_lower in CONTEXT_ROLES
                        ):
                            clean_roles[k_clean] = str(r_v)
                        else:
                            deferred.append((_heuristic_role_for_key(k_lower), str(r_v)))
                    # Heuristically mapped keys never overwrite an explicit role; on collision the
                    # value moves to the next free core slot instead of being silently lost.
                    used = {k.upper() if k.lower().startswith(":arg") else k.lower() for k in clean_roles}
                    for target, value in deferred:
                        slot = target
                        if slot in used:
                            free = [f":ARG{i}" for i in range(6) if f":ARG{i}" not in used]
                            if not free:
                                continue
                            slot = free[0] if target.startswith(":ARG") else target
                            if slot in used:
                                continue
                        clean_roles[slot] = value
                        used.add(slot)
                    ev["roles"] = clean_roles

                tc = ev.get("time_context")
                if tc is not None:
                    if not isinstance(tc, (dict, str, int, float)):
                        ev["time_context"] = None
                    elif isinstance(tc, dict):
                        raw = tc.get("raw_expression") or tc.get("expression") or tc.get("text") or tc.get("time")
                        if not raw and tc.get("start_year") is None and tc.get("end_year") is None:
                            ev["time_context"] = None
                        else:
                            tc["raw_expression"] = str(raw).strip() if raw is not None else ""

                sc = ev.get("spatial_context")
                if sc is not None:
                    if not isinstance(sc, (dict, str)):
                        ev["spatial_context"] = None
                    elif isinstance(sc, dict):
                        loc = sc.get("location_name") or sc.get("location") or sc.get("name") or sc.get("place")
                        if not loc:
                            ev["spatial_context"] = None
                        else:
                            sc["location_name"] = str(loc).strip()
                    elif isinstance(sc, str) and not sc.strip():
                        ev["spatial_context"] = None

        return v

    @field_validator("events", "discourse", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        return v if v is not None else []