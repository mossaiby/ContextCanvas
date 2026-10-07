"""Embedded KùzuDB-backed Context Graph with context envelopes, entity resolution and evidence retrieval."""
from __future__ import annotations

import gc
import os
import shutil
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import kuzu
import networkx as nx

from semantics.propbank import GLOBAL_CATALOG
from semantics.schema import DiscourseRelation, ExtractedEvent
from semantics.frames import IDENTITY_FRAMES
from semantics.text_utils import content_stems, entity_match_key, normalize_surface, slugify

_ROLE_ORDER = {f":ARG{i}": i for i in range(6)}


def _role_sort_key(role: str) -> Tuple[int, str]:
    if role in _ROLE_ORDER:
        return (_ROLE_ORDER[role], role)
    if role.upper().startswith(":ARGM"):
        return (10, role)
    return (20, role)


class ContextGraph:
    """Property graph storing events, argument bindings, and explicit context envelopes.

    KùzuDB is the system of record; a NetworkX mirror serves traversal. The mirror is rebuilt
    from KùzuDB on open, so a persisted database remains queryable across sessions.
    """

    def __init__(
        self,
        db_path: str = "context_canvas_kuzu",
        hub_degree: int = 25,
        render_default_provenance: bool = False,
        render_role_labels: bool = True,
        render_envelopes: bool = True,
        triples_mode: bool = False,
    ):
        self.db_path = db_path
        self.hub_degree = hub_degree
        self.render_default_provenance = render_default_provenance
        self.render_role_labels = render_role_labels
        self.render_envelopes = render_envelopes
        self.triples_mode = triples_mode
        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.kuzu_db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.kuzu_db)
        self.mirror = nx.MultiDiGraph()
        self._entity_index: Dict[str, str] = {}      # match key -> entity id
        self._used_entity_ids: Set[str] = set()
        self._event_signatures: Dict[Tuple, str] = {}
        self._event_counter = 0
        self._init_schema()
        self._load_mirror_from_db()

    # ------------------------------------------------------------------ lifecycle

    def _init_schema(self) -> None:
        tables = [
            "CREATE NODE TABLE Entity(id STRING, name STRING, PRIMARY KEY (id));",
            "CREATE NODE TABLE Event(id STRING, lemma STRING, sense_id STRING, PRIMARY KEY (id));",
            "CREATE NODE TABLE TimeContext(id STRING, raw_expr STRING, start_year INT64, end_year INT64, PRIMARY KEY (id));",
            "CREATE NODE TABLE PlaceContext(id STRING, location_name STRING, parent_region STRING, PRIMARY KEY (id));",
            "CREATE NODE TABLE EpistemicContext(id STRING, source STRING, confidence DOUBLE, is_speculative BOOLEAN, PRIMARY KEY (id));",
            "CREATE REL TABLE ArgumentRole(FROM Event TO Entity, role STRING);",
            "CREATE REL TABLE InTemporalContext(FROM Event TO TimeContext);",
            "CREATE REL TABLE InSpatialContext(FROM Event TO PlaceContext);",
            "CREATE REL TABLE InEpistemicContext(FROM Event TO EpistemicContext);",
            "CREATE REL TABLE DiscourseLink(FROM Event TO Event, relation STRING);",
        ]
        for ddl in tables:
            try:
                self.conn.execute(ddl)
            except Exception:
                pass  # table already exists

    def _rows(self, query: str) -> Iterable[List[Any]]:
        try:
            result = self.conn.execute(query)
        except Exception:
            return []
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def _load_mirror_from_db(self) -> None:
        """Rebuilds the traversal mirror and indexes from the persisted database."""
        for ev_id, lemma, sense in self._rows("MATCH (e:Event) RETURN e.id, e.lemma, e.sense_id"):
            self._add_event_node(ev_id, lemma, sense, source="")
        for ev_id, role, ent_id, name in self._rows(
            "MATCH (e:Event)-[r:ArgumentRole]->(n:Entity) RETURN e.id, r.role, n.id, n.name"
        ):
            self._add_entity_node(ent_id, name)
            self.mirror.add_edge(ev_id, ent_id, role=role, edge_type="role")
        for ev_id, t_id, raw, s_yr, e_yr in self._rows(
            "MATCH (e:Event)-[:InTemporalContext]->(t:TimeContext) RETURN e.id, t.id, t.raw_expr, t.start_year, t.end_year"
        ):
            ctx = {"raw_expression": raw, "start_year": None if s_yr == -1 else s_yr, "end_year": None if e_yr == -1 else e_yr}
            self.mirror.add_node(t_id, node_type="time", context=ctx)
            self.mirror.add_edge(ev_id, t_id, edge_type="temporal_context")
        for ev_id, p_id, loc, parent in self._rows(
            "MATCH (e:Event)-[:InSpatialContext]->(p:PlaceContext) RETURN e.id, p.id, p.location_name, p.parent_region"
        ):
            self.mirror.add_node(p_id, node_type="place", context={"location_name": loc, "parent_region": parent or None})
            self.mirror.add_edge(ev_id, p_id, edge_type="spatial_context")
        for ev_id, ec_id, src, conf, spec in self._rows(
            "MATCH (e:Event)-[:InEpistemicContext]->(c:EpistemicContext) RETURN e.id, c.id, c.source, c.confidence, c.is_speculative"
        ):
            self.mirror.add_node(ec_id, node_type="epistemic", context={"source": src, "confidence": conf, "is_speculative": spec})
            self.mirror.add_edge(ev_id, ec_id, edge_type="epistemic_context")
            if self.mirror.has_node(ev_id):
                self.mirror.nodes[ev_id]["source"] = src
        for src_id, tgt_id, rel in self._rows(
            "MATCH (a:Event)-[r:DiscourseLink]->(b:Event) RETURN a.id, b.id, r.relation"
        ):
            self.mirror.add_edge(src_id, tgt_id, relation=rel, edge_type="discourse")
        for ev in self.event_ids():
            self._event_signatures.setdefault(self._signature_from_mirror(ev), ev)

    def close(self) -> None:
        if getattr(self, "conn", None) is not None:
            close = getattr(self.conn, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self.conn = None
        if getattr(self, "kuzu_db", None) is not None:
            close = getattr(self.kuzu_db, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self.kuzu_db = None
        gc.collect()

    def reset(self) -> None:
        self.close()
        if os.path.exists(self.db_path):
            if os.path.isdir(self.db_path):
                shutil.rmtree(self.db_path, ignore_errors=True)
            else:
                try:
                    os.remove(self.db_path)
                except OSError:
                    pass
        parent = os.path.dirname(os.path.abspath(self.db_path))
        base = os.path.basename(self.db_path)
        if os.path.exists(parent):
            for fname in os.listdir(parent):
                if fname.startswith(base) and fname != base:
                    p = os.path.join(parent, fname)
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
        self.__init__(
            self.db_path,
            hub_degree=self.hub_degree,
            render_default_provenance=self.render_default_provenance,
            render_role_labels=self.render_role_labels,
            render_envelopes=self.render_envelopes,
            triples_mode=self.triples_mode,
        )

    # ------------------------------------------------------------------ mirror helpers

    def _add_event_node(self, ev_id: str, lemma: str, sense_id: str, source: str) -> None:
        self._event_counter += 1
        self.mirror.add_node(ev_id, node_type="event", lemma=lemma, sense_id=sense_id, order=self._event_counter, source=source)

    def _add_entity_node(self, ent_id: str, name: str) -> None:
        if not self.mirror.has_node(ent_id):
            self.mirror.add_node(ent_id, node_type="entity", name=name)
        self._used_entity_ids.add(ent_id)
        self._entity_index.setdefault(entity_match_key(name), ent_id)

    def event_ids(self) -> List[str]:
        return [n for n, d in self.mirror.nodes(data=True) if d.get("node_type") == "event"]

    def entity_names(self) -> List[str]:
        return [d["name"] for _, d in self.mirror.nodes(data=True) if d.get("node_type") == "entity" and d.get("name")]

    def entity_degree(self, entity_name: str) -> int:
        """Number of events an entity participates in (0 if unknown)."""
        ent = self.resolve_entity_id(entity_name)
        return len(self._entity_events(ent)) if ent else 0

    def has_event(self, ev_id: str) -> bool:
        return self.mirror.has_node(ev_id) and self.mirror.nodes[ev_id].get("node_type") == "event"

    def event_source(self, ev_id: str) -> str:
        return self.mirror.nodes[ev_id].get("source", "") if self.mirror.has_node(ev_id) else ""

    def _event_roles(self, ev_id: str) -> List[Tuple[str, str]]:
        """(role, entity_id) pairs of an event."""
        return [
            (d.get("role", ""), tgt)
            for _, tgt, d in self.mirror.out_edges(ev_id, data=True)
            if d.get("edge_type") == "role"
        ]

    def _entity_events(self, ent_id: str) -> List[Tuple[str, str]]:
        """(event_id, role) pairs in which an entity participates."""
        return [
            (src, d.get("role", ""))
            for src, _, d in self.mirror.in_edges(ent_id, data=True)
            if d.get("edge_type") == "role"
        ]

    def _context_of(self, ev_id: str, edge_type: str) -> List[Dict[str, Any]]:
        return [
            self.mirror.nodes[tgt].get("context", {})
            for _, tgt, d in self.mirror.out_edges(ev_id, data=True)
            if d.get("edge_type") == edge_type
        ]

    # ------------------------------------------------------------------ entity resolution

    def resolve_entity_id(self, entity_name: str) -> Optional[str]:
        """Read-only lookup of an existing entity id (never allocates)."""
        if not entity_name:
            return None
        return self._entity_index.get(entity_match_key(entity_name))

    def get_canonical_entity_id(self, entity_name: str) -> str:
        """Existing id for the name, or the id that would be allocated for it."""
        existing = self.resolve_entity_id(entity_name)
        if existing:
            return existing
        return self._allocate_entity_id(entity_match_key(entity_name), set())

    def _allocate_entity_id(self, key: str, reserved: Set[str]) -> str:
        base = "ent_" + slugify(key or "entity")
        candidate, n = base, 1
        while candidate in self._used_entity_ids or candidate in reserved:
            n += 1
            candidate = f"{base}_{n}"
        return candidate

    def _role_bindings(self, event: ExtractedEvent) -> List[Tuple[str, str]]:
        """Single source of truth for an event's (role, entity surface) bindings.

        The spatial envelope location is bridged into the entity layer as a ':location'
        argument unless the same entity is already bound to another role.
        """
        bindings = [(role, name) for role, name in event.roles.items() if name and str(name).strip()]
        sc = event.spatial_context
        if sc and sc.location_name:
            loc_key = entity_match_key(sc.location_name)
            if loc_key and loc_key not in {entity_match_key(n) for _, n in bindings} and ":location" not in event.roles:
                bindings.append((":location", sc.location_name))
        return bindings

    def _signature(self, event: ExtractedEvent, bindings: List[Tuple[str, str]]) -> Tuple:
        tc = event.time_context
        time_key = (tc.start_year, tc.end_year) if tc and tc.raw_expression else None
        return (
            event.sense_id,
            frozenset((role, entity_match_key(name)) for role, name in bindings),
            time_key,
        )

    def _signature_from_mirror(self, ev_id: str) -> Tuple:
        names = {eid: self.mirror.nodes[eid].get("name", "") for _, eid in self._event_roles(ev_id)}
        times = self._context_of(ev_id, "temporal_context")
        time_key = (times[0].get("start_year"), times[0].get("end_year")) if times else None
        return (
            self.mirror.nodes[ev_id].get("sense_id", ""),
            frozenset((role, entity_match_key(names[eid])) for role, eid in self._event_roles(ev_id)),
            time_key,
        )

    # ------------------------------------------------------------------ writes

    def insert_event(self, event: ExtractedEvent) -> str:
        """Inserts an event with its bindings and envelopes in one transaction.

        Semantically identical assertions (same sense, same role/entity bindings, same time) are
        stored once; the id of the existing event is returned for the duplicate.
        """
        bindings = self._role_bindings(event)
        signature = self._signature(event, bindings)
        if signature in self._event_signatures:
            return self._event_signatures[signature]

        ev_id = event.temp_id
        if self.mirror.has_node(ev_id):
            raise ValueError(f"Event id already exists in the graph: {ev_id}")

        planned: Dict[str, Tuple[str, str]] = {}   # key -> (entity id, display name)
        reserved: Set[str] = set()
        resolved: List[Tuple[str, str, str]] = []  # (role, entity id, display name)
        for role, raw_name in bindings:
            key = entity_match_key(raw_name)
            display = normalize_surface(raw_name)
            if key in planned:
                ent_id = planned[key][0]
            else:
                ent_id = self._entity_index.get(key) or self._allocate_entity_id(key, reserved)
                reserved.add(ent_id)
                planned[key] = (ent_id, display)
            resolved.append((role, ent_id, display))

        statements: List[Tuple[str, Dict[str, Any]]] = [(
            "CREATE (e:Event {id: $ev_id, lemma: $lemma, sense_id: $sense_id})",
            {"ev_id": ev_id, "lemma": event.lemma, "sense_id": event.sense_id},
        )]
        for ent_id, display in planned.values():
            statements.append((
                "MERGE (n:Entity {id: $ent_id}) ON CREATE SET n.name = $name",
                {"ent_id": ent_id, "name": display},
            ))
        for role, ent_id, _ in resolved:
            statements.append((
                "MATCH (e:Event {id: $ev_id}), (n:Entity {id: $ent_id}) CREATE (e)-[:ArgumentRole {role: $role}]->(n)",
                {"ev_id": ev_id, "ent_id": ent_id, "role": role},
            ))

        tc = event.time_context
        if tc and tc.raw_expression:
            statements.append((
                "CREATE (t:TimeContext {id: $t_id, raw_expr: $raw_expr, start_year: $start_year, end_year: $end_year})",
                {
                    "t_id": f"time_{ev_id}", "raw_expr": tc.raw_expression,
                    "start_year": tc.start_year if tc.start_year is not None else -1,
                    "end_year": tc.end_year if tc.end_year is not None else -1,
                },
            ))
            statements.append((
                "MATCH (e:Event {id: $ev_id}), (t:TimeContext {id: $t_id}) CREATE (e)-[:InTemporalContext]->(t)",
                {"ev_id": ev_id, "t_id": f"time_{ev_id}"},
            ))

        sc = event.spatial_context
        if sc and sc.location_name:
            statements.append((
                "CREATE (p:PlaceContext {id: $p_id, location_name: $location_name, parent_region: $parent_region})",
                {"p_id": f"place_{ev_id}", "location_name": sc.location_name, "parent_region": sc.parent_region or ""},
            ))
            statements.append((
                "MATCH (e:Event {id: $ev_id}), (p:PlaceContext {id: $p_id}) CREATE (e)-[:InSpatialContext]->(p)",
                {"ev_id": ev_id, "p_id": f"place_{ev_id}"},
            ))

        ec = event.epistemic_context
        statements.append((
            "CREATE (ep:EpistemicContext {id: $ec_id, source: $source, confidence: $confidence, is_speculative: $is_speculative})",
            {"ec_id": f"epistemic_{ev_id}", "source": ec.source, "confidence": ec.confidence, "is_speculative": ec.is_speculative},
        ))
        statements.append((
            "MATCH (e:Event {id: $ev_id}), (ep:EpistemicContext {id: $ec_id}) CREATE (e)-[:InEpistemicContext]->(ep)",
            {"ev_id": ev_id, "ec_id": f"epistemic_{ev_id}"},
        ))

        self.conn.execute("BEGIN TRANSACTION;")
        try:
            for query, params in statements:
                self.conn.execute(query, params)
            self.conn.execute("COMMIT;")
        except Exception as exc:
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise RuntimeError(f"Database write failed for event {ev_id}: {exc}") from exc

        # Mirror is only mutated after a successful commit, from the same resolved bindings.
        self._add_event_node(ev_id, event.lemma, event.sense_id, source=ec.source)
        for key, (ent_id, display) in planned.items():
            if not self.mirror.has_node(ent_id):
                self.mirror.add_node(ent_id, node_type="entity", name=display)
            self._used_entity_ids.add(ent_id)
            self._entity_index.setdefault(key, ent_id)
        for role, ent_id, _ in resolved:
            self.mirror.add_edge(ev_id, ent_id, role=role, edge_type="role")
        if tc and tc.raw_expression:
            self.mirror.add_node(f"time_{ev_id}", node_type="time", context=tc.model_dump())
            self.mirror.add_edge(ev_id, f"time_{ev_id}", edge_type="temporal_context")
        if sc and sc.location_name:
            self.mirror.add_node(f"place_{ev_id}", node_type="place", context=sc.model_dump())
            self.mirror.add_edge(ev_id, f"place_{ev_id}", edge_type="spatial_context")
        self.mirror.add_node(f"epistemic_{ev_id}", node_type="epistemic", context=ec.model_dump())
        self.mirror.add_edge(ev_id, f"epistemic_{ev_id}", edge_type="epistemic_context")
        self._event_signatures[signature] = ev_id
        return ev_id

    def insert_discourse_relation(self, rel: DiscourseRelation) -> None:
        """Inserts an inter-event discourse edge between two existing events."""
        if not (self.has_event(rel.source_id) and self.has_event(rel.target_id)):
            raise ValueError(f"Discourse relation references unknown event(s): {rel.source_id} -> {rel.target_id}")
        self.conn.execute("BEGIN TRANSACTION;")
        try:
            self.conn.execute(
                "MATCH (e1:Event {id: $src_id}), (e2:Event {id: $tgt_id}) CREATE (e1)-[:DiscourseLink {relation: $relation}]->(e2)",
                {"src_id": rel.source_id, "tgt_id": rel.target_id, "relation": rel.relation},
            )
            self.conn.execute("COMMIT;")
        except Exception as exc:
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise RuntimeError(f"Failed to persist discourse relation {rel}: {exc}") from exc
        self.mirror.add_edge(rel.source_id, rel.target_id, relation=rel.relation, edge_type="discourse")

    # ------------------------------------------------------------------ retrieval

    def _explore(self, anchor_ids: List[str], max_hops: int) -> Tuple[Dict[str, int], Dict[str, Tuple], Dict[str, int]]:
        """Breadth-first exploration over entity -> event -> entity steps.

        * Identity relations (alias.01) are crossed at zero cost: both names denote one entity.
        * Entities other than the anchors whose degree exceeds `hub_degree` are reached but not
          expanded, so generic hubs ("United States", a year) do not flood the evidence.
        Returns entity distances, entity parent pointers and event distances.
        """
        dist: Dict[str, int] = {a: 0 for a in anchor_ids}
        parent: Dict[str, Tuple] = {a: () for a in anchor_ids}
        event_dist: Dict[str, int] = {}
        frontier = deque(anchor_ids)
        while frontier:
            ent = frontier.popleft()
            d = dist[ent]
            events = self._entity_events(ent)
            if d > 0 and len(events) > self.hub_degree:
                continue
            for ev, role_in in events:
                identity = self.mirror.nodes[ev].get("sense_id") in IDENTITY_FRAMES
                if d >= max_hops and not identity:
                    continue
                event_dist.setdefault(ev, d)
                step = 0 if identity else 1
                for role_out, other in self._event_roles(ev):
                    if other not in dist or dist[other] > d + step:
                        dist[other] = d + step
                        parent[other] = (ent, role_in, ev, role_out)
                        if step == 0:
                            frontier.appendleft(other)
                        else:
                            frontier.append(other)
        return dist, parent, event_dist

    def _readable_path(self, ent_id: str, parent: Dict[str, Tuple]) -> str:
        parts: List[str] = [self.mirror.nodes[ent_id].get("name", ent_id)]
        cur = ent_id
        while parent.get(cur):
            prev, role_in, ev, role_out = parent[cur]
            if self.triples_mode:
                # Same information content as the triple rendering: predicate only, no roles.
                parts.append(self.mirror.nodes[ev].get("lemma", "related"))
            else:
                sense = self.mirror.nodes[ev].get("sense_id", "")
                parts.append(f"{role_in.lstrip(':')}<-[{sense}]->{role_out.lstrip(':')}")
            parts.append(self.mirror.nodes[prev].get("name", prev))
            cur = prev
        return " | ".join(reversed(parts))

    def _render_triples(self, ev: str) -> List[str]:
        """Ablation: the same event as plain (subject, predicate, object) triples, without role
        meanings or envelopes - the representation of a conventional triple-based KG."""
        lemma = self.mirror.nodes[ev].get("lemma", "event")
        roles = sorted(self._event_roles(ev), key=lambda x: _role_sort_key(x[0]))
        names = [self.mirror.nodes[e].get("name", e) for _, e in roles]
        if len(names) < 2:
            return [f"- ({names[0] if names else '?'}, {lemma}, -)"]
        return [f"- ({names[0]}, {lemma}, {other})" for other in names[1:]]

    def _render_event(self, ev: str) -> List[str]:
        if self.triples_mode:
            return self._render_triples(ev)
        data = self.mirror.nodes[ev]
        lemma, sense = data.get("lemma", "event"), data.get("sense_id", "")
        args = []
        for role, ent_id in sorted(self._event_roles(ev), key=lambda x: _role_sort_key(x[0])):
            label = GLOBAL_CATALOG.role_description(sense, role) if self.render_role_labels else ""
            role_txt = role.lstrip(":") + (f"[{label}]" if label else "")
            args.append(f"{role_txt}={self.mirror.nodes[ent_id].get('name', ent_id)}")
        lines = [f"- [{ev}] {lemma} ({sense}): " + ("; ".join(args) if args else "(no arguments)")]
        if not self.render_envelopes:
            return lines

        for t in self._context_of(ev, "temporal_context"):
            raw = t.get("raw_expression", "")
            s_yr, e_yr = t.get("start_year"), t.get("end_year")
            if s_yr not in (None, -1) and e_yr not in (None, -1):
                interval = f"[{s_yr}-{e_yr}]" if s_yr != e_yr else f"[{s_yr}]"
                lines.append(f"  Temporal Scope: {raw} {interval}".rstrip())
            elif raw:
                lines.append(f"  Temporal Scope: {raw}")
        for p in self._context_of(ev, "spatial_context"):
            loc, region = p.get("location_name", ""), p.get("parent_region", "")
            if loc:
                lines.append(f"  Spatial Scope: {loc} ({region})" if region else f"  Spatial Scope: {loc}")
        for c in self._context_of(ev, "epistemic_context"):
            conf = c.get("confidence", 1.0)
            spec = c.get("is_speculative", False)
            if conf >= 1.0 and not spec and not self.render_default_provenance:
                continue  # default envelope carries no information for the reader; provenance stays in bundle["sources"]
            status = "[SPECULATIVE]" if spec or conf < 0.60 else "[VERIFIED]"
            lines.append(f"  Epistemic Envelope: Source='{c.get('source', 'Unattributed')}', Certainty={int(round(conf * 100))}% {status}")
        for _, target, d in self.mirror.out_edges(ev, data=True):
            if d.get("edge_type") == "discourse":
                lines.append(f"  Discourse: --{d.get('relation', '')}--> [{target}]")
        return lines

    def _event_relevance(self, ev: str, question_stems: set) -> int:
        """Overlap between the question and the event's frame name and role meanings."""
        if not question_stems:
            return 0
        sense = self.mirror.nodes[ev].get("sense_id", "")
        words = [self.mirror.nodes[ev].get("lemma", ""), sense.split(".")[0]]
        for role, _ in self._event_roles(ev):
            words.append(GLOBAL_CATALOG.role_description(sense, role))
        return len(question_stems & content_stems(" ".join(words)))

    def build_evidence(
        self,
        anchor_names: List[str],
        max_hops: int = 3,
        max_events: int = 40,
        max_chars: Optional[int] = None,
        max_candidates: int = 10,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collects a bounded evidence subgraph around one or more anchors.

        Events are ordered by distance from the anchors and, within a distance, by relevance to
        the question. Candidates are drawn in turns across distances so multi-hop answers are
        not crowded out by the anchor's immediate neighbours.
        """
        anchor_ids = list(dict.fromkeys(a for a in (self.resolve_entity_id(n) for n in anchor_names) if a))
        empty = {"text": "", "event_ids": [], "candidates": [], "anchors": [], "sources": [], "entity_names": []}
        if not anchor_ids:
            return empty
        dist, parent, event_dist = self._explore(anchor_ids, max_hops)

        for ev in list(event_dist):
            for _, tgt, d in self.mirror.out_edges(ev, data=True):
                if d.get("edge_type") == "discourse":
                    event_dist.setdefault(tgt, event_dist[ev] + 1)
        if not event_dist:
            return {**empty, "anchors": [self.mirror.nodes[a]["name"] for a in anchor_ids]}

        q_stems = content_stems(question or "")
        relevance = {ev: self._event_relevance(ev, q_stems) for ev in event_dist}
        ordered = sorted(event_dist, key=lambda ev: (event_dist[ev], -relevance[ev], self.mirror.nodes[ev].get("order", 0)))

        lines = ["=== CONTEXT GRAPH EVIDENCE ==="]
        used_chars = len(lines[0])
        rendered: List[str] = []
        for ev in ordered[:max_events]:
            block = self._render_event(ev)
            size = sum(len(l) + 1 for l in block)
            if max_chars is not None and rendered and used_chars + size > max_chars:
                break
            lines.extend(block)
            used_chars += size
            rendered.append(ev)

        rendered_set = set(rendered)
        rendered_entities = {eid for ev in rendered for _, eid in self._event_roles(ev)}
        by_distance: Dict[int, List[str]] = {}
        for e in rendered_entities:
            if e in anchor_ids or e not in dist:
                continue
            by_distance.setdefault(dist[e], []).append(e)
        for d_key, ents in by_distance.items():
            ents.sort(key=lambda e: (
                -(relevance.get(parent[e][2], 0) if parent.get(e) else 0),
                parent[e][2] not in rendered_set if parent.get(e) else True,
                self.mirror.nodes[e].get("name", ""),
            ))
        candidate_ids: List[str] = []
        queues = [by_distance[k] for k in sorted(by_distance)]
        while len(candidate_ids) < max_candidates and any(queues):
            for q in queues:
                if q and len(candidate_ids) < max_candidates:
                    candidate_ids.append(q.pop(0))
        candidate_ids.sort(key=lambda e: dist[e])

        if candidate_ids:
            lines.append("")
            lines.append("=== DISCOVERED PATH CANDIDATES ===")
            for e in candidate_ids:
                lines.append(f"- Candidate: {self.mirror.nodes[e]['name']} (Path: {self._readable_path(e, parent)})")

        sources = sorted({self.event_source(ev) for ev in rendered if self.event_source(ev)})
        return {
            "text": "\n".join(lines),
            "event_ids": rendered,
            "candidates": [self.mirror.nodes[e]["name"] for e in candidate_ids],
            "anchors": [self.mirror.nodes[a]["name"] for a in anchor_ids],
            "sources": sources,
            "entity_names": sorted({self.mirror.nodes[e]["name"] for e in rendered_entities}),
        }

    def find_candidate_answers_from_paths(self, start_entity_name: str, max_hops: int = 3) -> List[Tuple[str, List[str]]]:
        """Entities reachable from the anchor with their readable path (nearest first)."""
        anchor = self.resolve_entity_id(start_entity_name)
        if not anchor:
            return []
        dist, parent, _ = self._explore([anchor], max_hops)
        out = []
        for e in sorted((e for e in dist if e != anchor), key=lambda e: (dist[e], self.mirror.nodes[e].get("name", ""))):
            out.append((self.mirror.nodes[e].get("name", e), self._readable_path(e, parent).split(" | ")))
        return out

    def query_subgraph_blueprint(self, entity_name: str, max_hops: int = 3, max_events: int = 40) -> str:
        """Serialises the local evidence subgraph around one entity."""
        if not self.resolve_entity_id(entity_name):
            return "No matching context graph entries found."
        bundle = self.build_evidence([entity_name], max_hops=max_hops, max_events=max_events)
        return bundle["text"] or "No event assertions linked to this entity."