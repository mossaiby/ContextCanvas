"""Unified pipeline connecting LLM extraction, semantic repair, sense validation and Context Graph QA."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from engine.realizer import Realizer
from memory.context_graph import ContextGraph
from semantics.propbank import GLOBAL_CATALOG
from semantics.schema import (
    DiscourseRelation,
    EpistemicContext,
    ExtractedEvent,
    ExtractionPayload,
    SpatialContext,
    TemporalContext,
)
from semantics.text_utils import (
    capitalised_spans,
    chunk_text,
    entity_match_key,
    is_date_like,
    is_placeholder,
    mine_acronym_aliases,
    parse_possessive_association,
    match_question_entities,
    parse_bare_kin_noun,
    parse_kinship_phrase,
    split_conjoined_names,
    split_place,
)

QUESTION_STOPWORDS = {
    "what", "which", "where", "when", "whose", "who", "whom", "how", "why",
    "this", "that", "these", "those", "from", "with", "located", "place",
    "facility", "institution", "university", "institute", "laboratory", "lab",
    "entity", "administrative", "territorial", "owner", "ownership", "state",
    "country", "city", "county", "town", "village", "person", "company", "group",
    "name", "type", "kind", "title", "called", "known", "part",
    "first", "last", "second", "third", "many", "much", "time", "year",
}

GENERIC_ANAPHORA_TARGETS = {
    "the film", "the movie", "the album", "the song", "the record", "the single",
    "the complex", "the venue", "the facility", "the company", "the band", "the team", "the station",
    "the university", "the institute", "the book", "the novel", "the play", "the series", "the show",
    "this film", "this movie", "this album", "this song", "this company",
    "the organization", "the organisation", "the group", "the territory", "the state", "the municipality",
}

CORE_ROLES = [f":ARG{i}" for i in range(6)]
LOCATION_ROLES = {":location", ":ARGM-LOC"}

# Kinship lemmas, grouped by what the SUBJECT (first core argument) is relative to the others.
SUBJECT_IS_CHILD = {"be_child_of", "child", "child_of", "son", "daughter", "son_of", "daughter_of", "born_to"}
SUBJECT_IS_PARENT = {"parent", "father", "mother", "beget", "father_of", "mother_of", "parent_of"}
BIRTH_GIVING = {"have_child", "give_birth"}
SPOUSE_LEMMAS = {"marry", "wed", "spouse", "husband", "wife", "be_married", "married", "marriage", "married_to"}
SIBLING_LEMMAS = {"sibling", "brother", "sister", "be_sibling_of"}
COPULA_LEMMAS = {"be", "is", "was", "be_a", "be_an"}
BIRTH_LEMMAS = {"born", "bear", "be_born"}
LOCATE_LEMMAS = {"locate", "headquarter", "headquarters", "be_located", "situate"}


def _drop_self_relations(events: List[ExtractedEvent]) -> List[ExtractedEvent]:
    """No entity relates to itself. A kinship event between one person and themselves is dropped
    entirely. In any other event, an entity filling two core roles ("DeSoto Records founded
    DeSoto Records", a person "giving birth to" themselves) loses the agent role (ARG0) or the
    later role, so the rest of the fact (founded in 1989, born in Nice) is kept."""
    kept = []
    for ev in events:
        if ev.sense_id in {"parent.01", "marry.01", "sibling.01"}:
            a, b = ev.roles.get(":ARG0"), ev.roles.get(":ARG1")
            if a and b and entity_match_key(a) == entity_match_key(b):
                continue
        core = [r for r in ev.roles if re.fullmatch(r":ARG\d", r)]
        seen: Dict[str, str] = {}
        for role in sorted(core, key=lambda r: (r == ":ARG0", r)):    # ARG0 last: it is the one dropped
            key = entity_match_key(ev.roles[role])
            if key in seen:
                del ev.roles[role]
            else:
                seen[key] = role
        kept.append(ev)
    return kept


def _relation_event(relation: str, subject: str, other: str, template: ExtractedEvent, suffix: str) -> ExtractedEvent:
    """Builds a canonical relation event from (subject relation-of other)."""
    if relation == "child":
        lemma, sense, roles = "parent", "parent.01", {":ARG0": other, ":ARG1": subject}
    elif relation == "parent":
        lemma, sense, roles = "parent", "parent.01", {":ARG0": subject, ":ARG1": other}
    elif relation == "spouse":
        lemma, sense, roles = "marry", "marry.01", {":ARG0": subject, ":ARG1": other}
    else:
        lemma, sense, roles = "sibling", "sibling.01", {":ARG0": subject, ":ARG1": other}
    return ExtractedEvent(
        temp_id=f"{template.temp_id}_{suffix}",
        lemma=lemma,
        sense_id=sense,
        roles=roles,
        time_context=template.time_context,
        epistemic_context=template.epistemic_context.model_copy(),
    )


# Every component the paper claims matters can be switched off for ablation studies.
DEFAULT_ABLATIONS: Dict[str, bool] = {
    "date_migration": True,        # move pure dates from arguments into the temporal envelope
    "placeholder_filter": True,    # drop "unknown"/pronoun arguments
    "kinship_repair": True,        # kinship lemmas / relational noun phrases -> parent/marry/sibling
    "copula_normalization": True,  # subject/predicate copulas -> be.01 roles
    "place_hierarchy": True,       # "A, B, C" in locative slots -> containment chain
    "possessive_links": True,      # "X's Facility" -> affiliate.01
    "alias_mining": True,          # "Long Name (ABBR)" -> alias.01
    "role_labels": True,           # render PropBank role meanings in the evidence
    "envelopes": True,             # render temporal / spatial / epistemic envelopes and discourse links
    "candidates": True,            # list path candidates after the facts
    "relevance_ranking": True,     # order same-distance events by question overlap
    "hub_pruning": True,           # drop hub anchors when a specific anchor exists
    "triples_only": False,         # render events as plain (subject, predicate, object) triples
}


class ContextCanvasEngine:
    """End-to-end engine coordinating extraction, graph storage, and query answering."""

    def __init__(
        self,
        db_path: str = "context_canvas_kuzu",
        config_path: str = "config.json",
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.realizer = Realizer(config_path=config_path, overrides=config_overrides)
        cfg = getattr(self.realizer, "config", {}) or {}
        self.ablations: Dict[str, bool] = {**DEFAULT_ABLATIONS, **(cfg.get("ablations") or {})}
        unknown = set(self.ablations) - set(DEFAULT_ABLATIONS)
        if unknown:
            raise ValueError(f"Unknown ablation flag(s): {sorted(unknown)}")
        self.max_chunk_chars = int(cfg.get("max_chunk_chars", 900))
        self.max_hops = int(cfg.get("evidence_max_hops", 3))
        self.max_evidence_events = int(cfg.get("evidence_max_events", 40))
        self.max_anchors = int(cfg.get("max_anchors", 4))
        self.graph = ContextGraph(
            db_path=db_path,
            hub_degree=int(cfg.get("hub_degree", 25)),
            render_default_provenance=bool(cfg.get("render_default_provenance", False)),
            render_role_labels=self.ablations["role_labels"],
            render_envelopes=self.ablations["envelopes"],
            triples_mode=self.ablations["triples_only"],
        )
        self._event_counter = 0
        self.last_query_status: Dict[str, Any] = {}

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self.graph.close()

    def reset(self) -> None:
        self._event_counter = 0
        self.last_query_status = {}
        self.graph.reset()

    def _evidence_char_budget(self) -> int:
        """Bounds the evidence by the model context window (≈3.5 characters per token)."""
        ctx = int(getattr(self.realizer, "context_window_tokens", 8192))
        reserve = int(getattr(self.realizer, "output_reserve_tokens", 1024))
        prompt_overhead = 700
        return max(2000, int((ctx - reserve - prompt_overhead) * 3.5))

    # ------------------------------------------------------------------ repair

    @staticmethod
    def _extract_document_title(text: str) -> Optional[str]:
        """Fallback title heuristic for 'Title. Body' inputs when no title is supplied."""
        m = re.match(r"^([A-Z0-9][^\n]{0,120}?)\.\s+(?=[A-Z])", text.strip())
        if not m:
            return None
        raw_title = m.group(1).strip()
        clean_title = re.sub(r"\s*\([^)]*\)$", "", raw_title).strip()
        return clean_title or raw_title

    @staticmethod
    def _migrate_dates(event: ExtractedEvent) -> None:
        """Moves pure temporal expressions out of argument slots into the temporal envelope."""
        for role in list(event.roles):
            val = event.roles[role]
            if is_date_like(val):
                if not event.time_context or not event.time_context.raw_expression:
                    event.time_context = TemporalContext(raw_expression=val)
                del event.roles[role]

    @staticmethod
    def _drop_placeholders(event: ExtractedEvent) -> None:
        """Removes filler values ("unknown", "someone", pronouns) that would otherwise become a
        single hub entity connecting unrelated facts."""
        for role in list(event.roles):
            if is_placeholder(event.roles[role]):
                del event.roles[role]

    def _possessor_is_entity(self, possessor: str, context_text: str) -> bool:
        """A possessor counts as an entity if it is an abbreviation ("DLR"), is already in the
        graph, or occurs in the text as a complete name of its own (not as the first word of a
        longer name such as "Grant" in "Grant Green", and not only in the possessive)."""
        if re.fullmatch(r"[A-Z][A-Z0-9&.\-]{1,7}", possessor):
            return True
        if self.graph.resolve_entity_id(possessor):
            return True
        pattern = r"(?<![\w'’])" + re.escape(possessor) + r"(?!['’]s\b)(?!\s+[A-Z])(?!\w)"
        for m in re.finditer(pattern, context_text or ""):
            preceding = context_text[: m.start()].rstrip().split()
            if preceding and preceding[-1][:1].isupper() and not preceding[-1].endswith((".", ",", ";", ":")):
                continue  # trailing part of a longer capitalised name
            return True
        return False

    def _derive_possessive_associations(
        self, event: ExtractedEvent, doc_title: Optional[str] = None, context_text: str = ""
    ) -> List[ExtractedEvent]:
        """'DLR's Lander Control Center' -> affiliate.01(DLR, DLR's Lander Control Center).

        Titles and names that merely contain a possessive ("Grant's First Stand", "National
        Women's Caucus") are not ownership: the derivation requires the possessor to be an
        entity in its own right and skips the document's own title."""
        derived: List[ExtractedEvent] = []
        title_key = entity_match_key(doc_title) if doc_title else None
        for role, val in list(event.roles.items()):
            parsed = parse_possessive_association(val)
            if not parsed:
                continue
            if title_key and entity_match_key(val) == title_key:
                continue
            if not self._possessor_is_entity(parsed[0], context_text):
                continue
            derived.append(ExtractedEvent(
                temp_id=f"{event.temp_id}_pos{len(derived)}",
                lemma="affiliate",
                sense_id="affiliate.01",
                roles={":ARG0": parsed[0], ":ARG1": val},
                epistemic_context=event.epistemic_context.model_copy(),
            ))
        return derived

    @staticmethod
    def _resolve_anaphora(event: ExtractedEvent, doc_title: Optional[str]) -> None:
        if not doc_title:
            return
        for role, val in list(event.roles.items()):
            if val.strip().lower() in GENERIC_ANAPHORA_TARGETS:
                event.roles[role] = doc_title

    @staticmethod
    def _subject_role(event: ExtractedEvent) -> Optional[str]:
        for r in CORE_ROLES:
            if r in event.roles:
                return r
        return None

    def _expand_kinship(self, event: ExtractedEvent) -> Optional[List[ExtractedEvent]]:
        """Rewrites kinship expressed via lemmas or relational noun phrases into canonical,
        direction-explicit relation events. Returns None when the event is not kinship."""
        lemma = event.lemma.lower().strip().replace(" ", "_").replace("-", "_")
        core = [(r, event.roles[r]) for r in CORE_ROLES if r in event.roles]
        derived: List[ExtractedEvent] = []

        if lemma in SPOUSE_LEMMAS | SIBLING_LEMMAS | SUBJECT_IS_PARENT | SUBJECT_IS_CHILD and len(core) >= 2:
            subject = core[0][1]
            relation = (
                "spouse" if lemma in SPOUSE_LEMMAS else
                "sibling" if lemma in SIBLING_LEMMAS else
                "parent" if lemma in SUBJECT_IS_PARENT else "child"
            )
            others = [v for _, v in core[1:]] if relation == "child" else [core[1][1]]
            for i, other in enumerate(others):
                derived.append(_relation_event(relation, subject, other, event, f"k{i}"))
            return derived

        if lemma in BIRTH_GIVING and len(core) >= 2:
            parent = core[0][1]
            if len(core) >= 3:
                child, co_parent = core[2][1], core[1][1]
                derived.append(_relation_event("parent", parent, child, event, "k0"))
                derived.append(_relation_event("parent", co_parent, child, event, "k1"))
            else:
                derived.append(_relation_event("parent", parent, core[1][1], event, "k0"))
            return derived

        # Relational noun phrases inside arguments: "X be son of Y", "X be younger son, ARG2 Y".
        # Not applied to birth events: there the noun phrase names the person born.
        if lemma in BIRTH_LEMMAS:
            return None
        subj_role = self._subject_role(event)
        if not subj_role:
            return None
        subject = event.roles[subj_role]
        found = False
        for role, val in core:
            if role == subj_role:
                continue
            parsed = parse_kinship_phrase(val)
            if parsed:
                relation, targets = parsed
                for i, target in enumerate(targets):
                    derived.append(_relation_event(relation, subject, target, event, f"{role.strip(':').lower()}{i}"))
                found = True
                continue
            bare = parse_bare_kin_noun(val)
            if bare:
                target_roles = [r for r, _ in core if r not in (subj_role, role)]
                targets = [t for tr in target_roles for t in split_conjoined_names(event.roles[tr])]
                for i, target in enumerate(targets):
                    derived.append(_relation_event(bare, subject, target, event, f"b{i}"))
                    found = True
        if not found:
            return None
        if lemma in COPULA_LEMMAS:
            return derived            # the copula carried nothing but the relation
        return [event] + derived

    @staticmethod
    def _normalize_copula(event: ExtractedEvent) -> None:
        """Maps subject/predicate copulas onto be.01 (:ARG1 topic, :ARG2 comment)."""
        if event.lemma.lower() in COPULA_LEMMAS and ":ARG0" in event.roles and ":ARG2" not in event.roles:
            subject = event.roles.pop(":ARG0")
            if ":ARG1" in event.roles:
                event.roles[":ARG2"] = event.roles.pop(":ARG1")
            event.roles[":ARG1"] = subject
            event.lemma, event.sense_id = "be", "be.01"

    @staticmethod
    def _normalize_birth(event: ExtractedEvent) -> List[ExtractedEvent]:
        """Normalises birth events onto bear.02 (:ARG0 mother/parent, :ARG1 person born).

        Extractors write "X was born" with the person first (:ARG0). The remaining non-date
        argument is the birthplace if there is exactly one; two or more are the parents.
        Returns derived parent events (if any).
        """
        lemma = event.lemma.lower()
        if lemma not in BIRTH_LEMMAS:
            return []
        roles = dict(event.roles)
        if lemma == "bear":
            if event.sense_id != "bear.02":
                return []  # bear.01 (carry/endure) is not a birth
            person, parent = roles.pop(":ARG1", None), roles.pop(":ARG0", None)
        else:
            person, parent = roles.pop(":ARG0", None) or roles.pop(":ARG1", None), None
        remaining = [roles.pop(r) for r in CORE_ROLES if r in roles]
        new_roles: Dict[str, str] = {":ARG1": person} if person else {}
        if parent:
            new_roles[":ARG0"] = parent
        derived: List[ExtractedEvent] = []
        if len(remaining) == 1:
            place = remaining[0]
            if not event.spatial_context or not event.spatial_context.location_name:
                event.spatial_context = SpatialContext(location_name=place)
            elif place != event.spatial_context.location_name:
                new_roles[":ARGM-LOC"] = place
        elif len(remaining) >= 2 and person:
            for i, p in enumerate(remaining):
                derived.append(_relation_event("parent", p, person, event, f"p{i}"))
        new_roles.update(roles)  # modifiers / contextual roles
        event.roles = new_roles
        event.lemma, event.sense_id = "bear", "bear.02"
        return derived

    @staticmethod
    def _normalize_locate(event: ExtractedEvent) -> None:
        """locate.01: :ARG1 is the located thing, :ARG2 the location."""
        if event.lemma.lower() not in LOCATE_LEMMAS:
            return
        event.lemma, event.sense_id = "locate", "locate.01"
        roles = event.roles
        if ":ARG0" in roles and ":ARG1" not in roles:
            roles[":ARG1"] = roles.pop(":ARG0")
        elif ":ARG0" in roles and ":ARG1" in roles and ":ARG2" not in roles:
            roles[":ARG2"] = roles.pop(":ARG1")
            roles[":ARG1"] = roles.pop(":ARG0")
        if ":ARG2" not in roles:
            for loc_role in (":location", ":ARGM-LOC"):
                if loc_role in roles:
                    roles[":ARG2"] = roles.pop(loc_role)
                    break

    def _expand_place_hierarchies(self, event: ExtractedEvent) -> List[ExtractedEvent]:
        """'Canyon, Texas, United States' in a locative slot -> head 'Canyon' plus explicit
        containment events (Canyon in Texas, Texas in United States)."""
        chains: List[List[str]] = []
        locative_roles = set(LOCATION_ROLES)
        if event.sense_id == "locate.01":
            locative_roles.add(":ARG2")
        for role in list(event.roles):
            if role in locative_roles:
                parts = split_place(event.roles[role])
                if parts:
                    event.roles[role] = parts[0]
                    chains.append(parts)
        sc = event.spatial_context
        if sc and sc.location_name and sc.parent_region:
            parent_parts = split_place(sc.parent_region) or [sc.parent_region]
            chains.append([sc.location_name] + parent_parts)

        derived: List[ExtractedEvent] = []
        seen = set()
        for chain in chains:
            for inner, outer in zip(chain, chain[1:]):
                if (inner, outer) in seen:
                    continue
                seen.add((inner, outer))
                derived.append(ExtractedEvent(
                    temp_id=f"{event.temp_id}_loc{len(derived)}",
                    lemma="locate",
                    sense_id="locate.01",
                    roles={":ARG1": inner, ":ARG2": outer},
                    epistemic_context=event.epistemic_context.model_copy(),
                ))
        return derived

    def _finalize_sense_and_roles(self, event: ExtractedEvent, context_text: str) -> None:
        if event.sense_id not in GLOBAL_CATALOG.framesets:
            resolved = GLOBAL_CATALOG.disambiguate_sense(event.lemma, context_text)
            if resolved:
                event.sense_id = resolved.roleset_id
        event.roles = GLOBAL_CATALOG.validate_roles(event.sense_id, event.roles)

    def _prepare_events(
        self,
        event: ExtractedEvent,
        doc_title: Optional[str] = None,
        source: Optional[str] = None,
        context_text: str = "",
    ) -> List[ExtractedEvent]:
        """Applies all semantic repairs to one extracted event; may return several events."""
        if source and event.epistemic_context.source == "Unattributed":
            event.epistemic_context = event.epistemic_context.model_copy(update={"source": source})

        ab = self.ablations
        if ab["date_migration"]:
            self._migrate_dates(event)
        if ab["placeholder_filter"]:
            self._drop_placeholders(event)
        self._resolve_anaphora(event, doc_title)

        expanded = self._expand_kinship(event) if ab["kinship_repair"] else None
        events = _drop_self_relations(expanded if expanded is not None else [event])

        prepared: List[ExtractedEvent] = []
        for ev in events:
            if ab["copula_normalization"]:
                self._normalize_copula(ev)
            derived = self._normalize_birth(ev)
            self._normalize_locate(ev)
            if ab["place_hierarchy"]:
                derived += self._expand_place_hierarchies(ev)
            if ab["possessive_links"]:
                derived += self._derive_possessive_associations(ev, doc_title, context_text)
            for e in [ev] + derived:
                self._finalize_sense_and_roles(e, context_text or " ".join(e.roles.values()))
                if e.roles:
                    prepared.append(e)
        return prepared

    # ------------------------------------------------------------------ ingestion

    def _unique_event_id(self, proposed: str, taken: set) -> str:
        candidate = proposed or "ev"
        while self.graph.mirror.has_node(candidate) or candidate in taken:
            self._event_counter += 1
            candidate = f"{proposed or 'ev'}_{self._event_counter}"
        return candidate

    def ingest_text(
        self,
        text: str,
        source: str = "Document",
        title: Optional[str] = None,
        debug: bool = False,
    ) -> List[str]:
        """Extracts events chunk by chunk, repairs and validates them, and commits them."""
        doc_title = title or self._extract_document_title(text)
        chunks = chunk_text(text, self.max_chunk_chars)
        multi = len(chunks) > 1

        inserted_ids: List[str] = []
        for c_idx, chunk in enumerate(chunks):
            chunk_input = chunk
            if multi and doc_title and not chunk.startswith(doc_title):
                chunk_input = f"{doc_title}. {chunk}"
            payload: ExtractionPayload = self.realizer.extract_structured_context(chunk_input, source=source)

            id_map: Dict[str, str] = {}
            taken: set = set()
            for event in payload.events:
                original_id = event.temp_id or "ev"
                if multi:
                    event.temp_id = f"c{c_idx}_{original_id}"
                prepared = self._prepare_events(event, doc_title=doc_title, source=source, context_text=chunk)
                stored_for_original: Optional[str] = None
                for ev in prepared:
                    ev.temp_id = self._unique_event_id(ev.temp_id, taken)
                    taken.add(ev.temp_id)
                    try:
                        ev_id = self.graph.insert_event(ev)
                    except Exception as exc:
                        if debug:
                            print(f"      [INSERT FAILED] {ev.lemma}: {exc}")
                        continue
                    is_new = ev_id == ev.temp_id
                    if ev_id not in inserted_ids:
                        inserted_ids.append(ev_id)
                    stored_for_original = stored_for_original or ev_id
                    if debug and is_new:
                        print(f"        * [{ev.lemma} / {ev.sense_id}] {ev.roles}"
                              + (f" time={ev.time_context.raw_expression!r}" if ev.time_context else "")
                              + (f" place={ev.spatial_context.location_name!r}" if ev.spatial_context else ""))
                if stored_for_original:
                    id_map[original_id] = stored_for_original

            for long_name, acronym in (mine_acronym_aliases(chunk) if self.ablations["alias_mining"] else []):
                alias_ev = ExtractedEvent(
                    temp_id=self._unique_event_id(f"alias_c{c_idx}", taken),
                    lemma="alias",
                    sense_id="alias.01",
                    roles={":ARG0": long_name, ":ARG1": acronym},
                    epistemic_context=EpistemicContext(source=source),
                )
                taken.add(alias_ev.temp_id)
                try:
                    ev_id = self.graph.insert_event(alias_ev)
                    if ev_id not in inserted_ids:
                        inserted_ids.append(ev_id)
                    if debug:
                        print(f"        * [alias / alias.01] {long_name} = {acronym}")
                except Exception:
                    pass

            for rel in payload.discourse:
                src, tgt = id_map.get(rel.source_id), id_map.get(rel.target_id)
                if not src or not tgt or src == tgt:
                    continue
                try:
                    self.graph.insert_discourse_relation(DiscourseRelation(source_id=src, target_id=tgt, relation=rel.relation))
                except Exception:
                    continue
        return inserted_ids

    def ingest_event_direct(self, event: ExtractedEvent) -> str:
        """Inserts a pre-constructed event (plus any derived events) without LLM extraction."""
        prepared = self._prepare_events(event)
        first_id = ""
        taken: set = set()
        for i, ev in enumerate(prepared):
            if i > 0:
                ev.temp_id = self._unique_event_id(ev.temp_id, taken)
            taken.add(ev.temp_id)
            ev_id = self.graph.insert_event(ev)
            first_id = first_id or ev_id
        return first_id

    # ------------------------------------------------------------------ question answering

    def _find_candidate_entities(self, question: str) -> List[str]:
        """Anchors the question in the graph (see text_utils.match_question_entities)."""
        names = self.graph.entity_names()
        anchors = match_question_entities(question, names, stopwords=sorted(QUESTION_STOPWORDS))
        if anchors:
            return anchors

        # Fallback: an entity whose name contains ALL words of a multi-word capitalised question
        # span (in any order). Sharing one word (a surname) is not enough to anchor on.
        spans = [
            {t for t in re.findall(r"\w+", text.lower()) if t not in QUESTION_STOPWORDS and len(t) > 1}
            for _, _, text in capitalised_spans(question)
        ]
        spans = [sp for sp in spans if len(sp) >= 2]
        if not spans:
            return []
        hits = []
        for name in names:
            name_tokens = {w.lower() for w in re.findall(r"\w+", name)}
            if any(sp <= name_tokens for sp in spans):
                hits.append(name)
        hits.sort(key=lambda n: len(n.split()))
        return hits[: self.max_anchors]

    def _drop_hub_anchors(self, anchors: List[str]) -> List[str]:
        """With several anchors, drop generic hubs ("Serbia" in "Tihomir of Serbia"): their
        many facts would crowd out the evidence about the specific entity asked about."""
        if len(anchors) <= 1:
            return anchors
        specific = [a for a in anchors if self.graph.entity_degree(a) <= self.graph.hub_degree]
        return specific or anchors

    @staticmethod
    def _snap_to_evidence(response: str, entity_names: List[str]) -> str:
        """Returns the answer in the exact surface form used in the evidence, when the model's
        answer resolves to an evidence entity (e.g. 'Bombardier Inc' -> 'Bombardier Inc.')."""
        if not response.startswith("ANSWER:"):
            return response
        answer = response[len("ANSWER:"):].strip()
        key = entity_match_key(answer)
        for name in entity_names:
            if entity_match_key(name) == key:
                return f"ANSWER: {name}"
        return response

    def ask(self, question: str, target_entity: Optional[str] = None, debug: bool = False) -> str:
        """Answers a question from the bounded evidence subgraph around its anchor entities."""
        anchors = [target_entity] if target_entity else self._find_candidate_entities(question)
        if debug:
            print(f"\n    [DIAG] Matched Anchor Entities: {anchors}")
        if not anchors:
            self.last_query_status = {"stage": "graph_no_anchors", "anchors": []}
            return "STATUS: NOT_IN_EVIDENCE"

        matched_anchors = list(anchors)
        if self.ablations["hub_pruning"]:
            anchors = self._drop_hub_anchors(anchors)
        if debug and anchors != matched_anchors:
            print(f"    [DIAG] Anchors after hub pruning: {anchors}")
        bundle = self.graph.build_evidence(
            anchors[: self.max_anchors],
            max_hops=self.max_hops,
            max_events=self.max_evidence_events,
            max_chars=self._evidence_char_budget(),
            question=question if self.ablations["relevance_ranking"] else None,
            max_candidates=10 if self.ablations["candidates"] else 0,
        )
        if not bundle["event_ids"]:
            if debug:
                print("    [DIAG] Evidence: EMPTY")
            self.last_query_status = {"stage": "graph_empty_blueprint", "anchors": anchors}
            return "STATUS: NOT_IN_EVIDENCE"
        if debug:
            print(f"    [DIAG] Evidence Sent to LLM:\n{bundle['text']}\n")

        raw_response = self.realizer.answer_question(
            question, bundle["text"], candidate_entities=bundle["candidates"] or None
        )
        raw_response = self._snap_to_evidence(raw_response, bundle.get("entity_names", []))
        answer_trace = getattr(self.realizer, "last_answer_text", "")
        if debug:
            print(f"    [DIAG] LLM Output:\n{answer_trace}\n")
            print(f"    [DIAG] Parsed Response: \"{raw_response}\"")

        self.last_query_status = {
            "stage": "qa_refusal" if raw_response.startswith("STATUS:") else "qa_success",
            "anchors": anchors,
            "matched_anchors": matched_anchors,
            "candidate_entities": bundle["candidates"],
            "evidence_event_ids": bundle["event_ids"],
            "evidence_sources": bundle["sources"],
            "evidence_text": bundle["text"],
            "evidence_chars": len(bundle["text"]),
            "raw_response": raw_response,
            "llm_output": answer_trace,
        }
        return raw_response