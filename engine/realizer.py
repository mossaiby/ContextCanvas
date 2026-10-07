"""LLM realizer enforcing structured output validation, token limits, and timeouts."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import ValidationError

from semantics.frames import prompt_frame_lines
from semantics.schema import ExtractionPayload
from semantics.text_utils import _strip_sentence_period

ANSWER_PATTERN = re.compile(
    r"^\s*(?:FINAL[ _]ANSWER|ANSWER)\s*:\s*(?P<answer>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
STATUS_PATTERN = re.compile(r"^\s*(?:STATUS\s*:\s*)?NOT_IN_EVIDENCE\s*$", re.IGNORECASE | re.MULTILINE)

# An answer line counts as a refusal only if the WHOLE line is a refusal, so legitimate
# answers that merely contain such words ("Unknown Pleasures", "No Escape") survive.
REFUSAL_RE = re.compile(
    r"^(?:status\s*:\s*)?(?:not[_ ]in[_ ]evidence|unknown|unsupported|none|n/?a|not|no answer|"
    r"no (?:explicit )?evidence\b.*|(?:i )?(?:cannot|can't|can not) (?:be )?determined?\b.*|"
    r"there is no\b.*|(?:it is |this is )?not (?:explicitly )?(?:provided|stated|mentioned|specified|available)\b.*|"
    r"(?:the )?(?:evidence|context|graph) does not\b.*|i don'?t have\b.*|insufficient (?:evidence|information)\b.*)$",
    re.IGNORECASE,
)
PREAMBLE_RE = re.compile(
    r"^(?:based\s+on\s+the\s+(?:evidence|context|facts|graph)\s*,?\s*|the\s+answer\s+is\s*:?\s*|answer\s*:\s*)+",
    re.IGNORECASE,
)


def _clean_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    first_brace, last_brace = text.find("{"), text.rfind("}")
    candidate = text[first_brace:last_brace + 1] if 0 <= first_brace < last_brace else text
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"events": data}
    except Exception:
        pass
    return _salvage_truncated_json(text)


def _salvage_truncated_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Recovers the complete events of a JSON array that was cut off mid-way."""
    text = raw_text.strip()
    events_idx = text.find('"events"')
    if events_idx == -1:
        return None
    first_brace = text.find("{")
    if first_brace == -1 or first_brace > events_idx:
        return None
    last_brace = text.rfind("}")
    while last_brace > events_idx:
        candidate = text[first_brace: last_brace + 1].strip().rstrip(",")
        for closing in ["]}", "}", "]"]:
            try:
                parsed = json.loads(candidate + closing)
                if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
                    return parsed
            except Exception:
                continue
        last_brace = text.rfind("}", first_brace, last_brace)
    return None


def _message_fields(msg: Any) -> Dict[str, str]:
    fields = {"content": (getattr(msg, "content", "") or "").strip()}
    for field in ("thinking", "reasoning", "reasoning_content"):
        val = getattr(msg, field, None)
        if not (isinstance(val, str) and val.strip()) and isinstance(getattr(msg, "model_extra", None), dict):
            val = msg.model_extra.get(field)
        if isinstance(val, str) and val.strip():
            fields[field] = val.strip()
    return fields


def _extract_json_content(msg: Any) -> str:
    fields = _message_fields(msg)
    if fields["content"]:
        return fields["content"]
    for field in ("thinking", "reasoning", "reasoning_content"):
        if field in fields:
            return fields[field]
    return ""


def _extract_message_text(msg: Any) -> str:
    """Content first; reasoning is appended BEFORE content so the final answer line wins."""
    fields = _message_fields(msg)
    reasoning = "\n".join(dict.fromkeys(v for k, v in fields.items() if k != "content"))
    content = fields["content"]
    if content and ANSWER_PATTERN.search(content):
        return content
    return f"{reasoning}\n{content}".strip()


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) else 0


class Realizer:
    """LLM access for extraction and answering, with caching and usage accounting."""

    # Bump when an extraction prompt change should invalidate cached extractions.
    EXTRACTION_PROMPT_VERSION = "2026-10-06"

    def __init__(self, config_path: str = "config.json", overrides: Optional[Dict[str, Any]] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = _deep_merge(cfg, overrides)
        self.config: Dict[str, Any] = cfg

        # Extraction and answering may use different models: extraction is a high-volume
        # structured task (one call per paragraph chunk), answering is low-volume reasoning.
        # "model_name" is the default for both.
        default_model = cfg.get("model_name")
        self.extraction_model = cfg.get("extraction_model") or default_model
        self.answer_model = cfg.get("answer_model") or default_model
        if not self.extraction_model or not self.answer_model:
            raise ValueError("config needs 'model_name' or both 'extraction_model' and 'answer_model'")
        self.model_name = self.answer_model
        self.base_url = cfg["base_url"]
        self.api_key = cfg["api_key"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.context_window_tokens = int(cfg.get("context_window_tokens", 8192))
        self.output_reserve_tokens = int(cfg.get("output_reserve_tokens", 1024))
        self.max_events = int(cfg.get("max_events_per_extraction", 15))
        self.extract_timeout = float(cfg.get("extract_timeout_s", 60.0))
        self.answer_timeout = float(cfg.get("answer_timeout_s", 60.0))
        self.disable_thinking = bool(cfg.get("disable_thinking", False))
        # "best_supported" (default): answer from the best fact chain even when the question words a
        # relation differently; "strict": refuse unless every asked relation is stated. Report both.
        self.answer_policy = str(cfg.get("answer_policy", "best_supported")).lower()
        self.last_answer_text = ""
        self._last_call_cost: Dict[str, float] = {}
        cache_dir = cfg.get("extraction_cache_dir")
        self.extraction_cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.extraction_cache_dir:
            self.extraction_cache_dir.mkdir(parents=True, exist_ok=True)
        self.reset_usage()

        self.extra_body: Dict[str, Any] = {
            "num_ctx": self.context_window_tokens,
            "options": {"num_ctx": self.context_window_tokens},
        }
        if self.disable_thinking:
            self.extra_body["reasoning_effort"] = "none"
            self.extra_body["think"] = False

        # Retries are bounded explicitly: with the SDK default a hung endpoint multiplies the
        # per-request timeout and a single call can stall a run for many minutes.
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=max(self.extract_timeout, self.answer_timeout),
            max_retries=int(cfg.get("max_retries", 1)),
        )

    # ------------------------------------------------------------------ usage accounting

    def reset_usage(self) -> None:
        self.usage: Dict[str, Dict[str, float]] = {}

    def usage_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self.usage.items()}

    def usage_summary(self) -> Dict[str, float]:
        """Index-time (extraction) vs query-time (answering) cost of the work since reset."""
        def total(kinds, field):
            return sum(self.usage.get(k, {}).get(field, 0) for k in kinds)
        index, query = ("extract",), ("answer", "finalize")
        tok = lambda kinds: total(kinds, "prompt_tokens") + total(kinds, "completion_tokens")
        return {
            "index_tokens": tok(index), "query_tokens": tok(query), "total_tokens": tok(index) + tok(query),
            "index_seconds": total(index, "seconds"), "query_seconds": total(query, "seconds"),
            "seconds": total(index, "seconds") + total(query, "seconds"),
            "llm_calls": total(index + query, "calls"), "extract_cache_hits": total(index, "cache_hits"),
            "finalize_calls": total(("finalize",), "calls"),
        }

    def _bucket(self, kind: str) -> Dict[str, float]:
        return self.usage.setdefault(kind, {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})

    def _record(self, kind: str, response: Any, seconds: float) -> Dict[str, float]:
        u = self._bucket(kind)
        usage = getattr(response, "usage", None)
        delta = {
            "prompt_tokens": _int_or_zero(getattr(usage, "prompt_tokens", 0)),
            "completion_tokens": _int_or_zero(getattr(usage, "completion_tokens", 0)),
            "seconds": seconds,
        }
        u["calls"] += 1
        for k, v in delta.items():
            u[k] += v
        return delta

    def _record_cached(self, kind: str, original: Dict[str, float]) -> None:
        """A cache hit is charged the cost recorded when the result was first computed, so cost
        figures do not depend on the order in which systems were run."""
        u = self._bucket(kind)
        u["cache_hits"] += 1
        for k in ("prompt_tokens", "completion_tokens", "seconds"):
            u[k] += float(original.get(k, 0) or 0)

    def _chat(self, kind: str, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        model = self.extraction_model if kind == "extract" else self.answer_model
        response = self.client.chat.completions.create(model=model, **kwargs)
        self._last_call_cost = self._record(kind, response, time.perf_counter() - t0)
        return response

    # ------------------------------------------------------------------ extraction

    def _extraction_prompt(self) -> str:
        return (
            "You are an expert knowledge graph extraction engine.\n"
            f"Extract the relational facts stated in the text as events in valid JSON (at most {self.max_events} events). "
            "Prefer facts that link two named entities; do not split one fact into several events.\n\n"
            "MANDATORY JSON FORMAT:\n"
            "{\n"
            '  "events": [\n'
            '    {\n'
            '      "temp_id": "ev1",\n'
            '      "lemma": "verb_in_infinitive",\n'
            '      "sense_id": "lemma.01",\n'
            '      "roles": {":ARG0": "agent / subject", ":ARG1": "patient / object", ":ARG2": "secondary argument"},\n'
            '      "time_context": {"raw_expression": "date or period if stated"},\n'
            '      "spatial_context": {"location_name": "place if stated"}\n'
            '    }\n'
            '  ],\n'
            '  "discourse": [{"source_id": "ev1", "target_id": "ev2", "relation": ":before"}]\n'
            "}\n\n"
            "RULES:\n"
            "1. Resolve pronouns and generic nouns ('the film', 'he', 'the company') to the specific proper name.\n"
            "2. Role values are entity names only - never dates (dates go in time_context) and never whole clauses.\n"
            "3. Use these frames for common relations:\n"
            + "\n".join(prompt_frame_lines()) + "\n"
            "4. Express relations with the specific verb above instead of 'be' + a noun phrase "
            "('X is the son of Y' -> parent.01 {':ARG0': 'Y', ':ARG1': 'X'}).\n"
            "5. Omit time_context / spatial_context when not stated. Emit ONLY raw JSON, no prose or code fences."
        )

    def _cache_path(self, system_prompt: str, text: str) -> Optional[Path]:
        if not self.extraction_cache_dir:
            return None
        key = hashlib.sha256(
            json.dumps([self.EXTRACTION_PROMPT_VERSION, self.extraction_model, system_prompt, text], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.extraction_cache_dir / f"{key[:2]}" / f"{key}.json"

    def extract_structured_context(self, text: str, source: str = "Input Document") -> ExtractionPayload:
        """Extracts events into JSON matching the schema. The output is a pure function of the
        text (the source id is not shown to the model), so it can be cached and shared by every
        system variant that ingests the same paragraph."""
        system_prompt = self._extraction_prompt()
        cache_path = self._cache_path(system_prompt, text)
        if cache_path and cache_path.exists():
            try:
                entry = json.loads(cache_path.read_text(encoding="utf-8"))
                payload = ExtractionPayload.model_validate(entry["payload"])
                self._record_cached("extract", entry.get("cost", {}))
                return payload
            except Exception:
                pass
        spent = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

        user_prompt = f"Text: \"{text}\"\n\nEmit JSON:"
        extract_extra = dict(self.extra_body)
        extract_extra["think"] = False
        extract_extra["reasoning_effort"] = "none"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        for attempt in range(2):
            try:
                response = self._chat(
                    "extract",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048 if attempt == 0 else 1536,
                    timeout=self.extract_timeout,
                    response_format={"type": "json_object"},
                    extra_body=extract_extra,
                )
                raw_content = _extract_json_content(response.choices[0].message)
                for k in spent:
                    spent[k] += self._last_call_cost.get(k, 0)
            except Exception as e:
                print(f" [Extraction error (attempt {attempt + 1}): {e}]", end="", flush=True)
                if attempt == 0:
                    continue
                return ExtractionPayload()
            parsed = _clean_and_parse_json(raw_content)
            if parsed:
                try:
                    payload = ExtractionPayload.model_validate(parsed)
                    if cache_path:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps({"payload": parsed, "cost": spent, "model": self.extraction_model},
                                                         ensure_ascii=False), encoding="utf-8")
                        with open(self.extraction_cache_dir / "texts.jsonl", "a", encoding="utf-8") as idx:
                            idx.write(json.dumps({"key": cache_path.stem, "text": text, "model": self.extraction_model},
                                                 ensure_ascii=False) + "\n")
                    return payload
                except ValidationError:
                    pass
            messages = messages[:2] + [{
                "role": "user",
                "content": "Return ONLY valid JSON matching {'events': [{'temp_id': 'ev1', 'lemma': '...', "
                           "'sense_id': '....01', 'roles': {':ARG0': '...', ':ARG1': '...'}}]}.",
            }]
        return ExtractionPayload()

    # ------------------------------------------------------------------ answering

    @staticmethod
    def _clean_answer(candidate: str, question: str) -> Optional[str]:
        # Periods are not stripped blindly: they belong to abbreviations ("Inc.", "Jr.", "D.C.").
        candidate = candidate.strip().strip("\"'`*,;: ")
        candidate = re.sub(r"^(?:STATUS\s*:\s*)+", "", candidate, flags=re.IGNORECASE).strip()
        candidate = PREAMBLE_RE.sub("", candidate).strip().strip("\"'`*,;: ")
        candidate = _strip_sentence_period(candidate)
        if not candidate or candidate.startswith("<") or "entity name" in candidate.lower():
            return None
        if REFUSAL_RE.match(candidate):
            return None
        # "Stanley, North Dakota" -> "Stanley" unless the question asks for the enclosing region.
        if "," in candidate and not re.search(r"\b(state|country|province|region|county)\b", question, re.IGNORECASE):
            head, tail = candidate.split(",", 1)
            if len(head.split()) <= 3 and tail.strip()[:1].isupper():
                candidate = head.strip()
        return candidate

    def _policy_rule(self, evidence_kind: str) -> str:
        source = {"graph": "facts", "text": "passages", "none": "what you know"}[evidence_kind]
        if evidence_kind == "none":
            return ("4. Answer from your own knowledge. If you do not know the answer, answer NOT_IN_EVIDENCE.\n")
        if self.answer_policy == "strict":
            return (f"4. Answer only if the {source} state every relation the question asks about; "
                    "otherwise answer NOT_IN_EVIDENCE.\n")
        return (
            "4. First identify the TYPE of answer requested (a person, a place, an organisation, ...). The question "
            f"may word a relation differently from the {source}, or a link may only be implied by them. Follow the "
            f"chain of {source} from the question entity and give the best-supported entity of the requested type. "
            f"Answer NOT_IN_EVIDENCE only if nothing in the {source} reaches any entity of that type.\n"
        )

    def _system_prompt(self, evidence_kind: str) -> str:
        if evidence_kind == "graph":
            intro = (
                "You are a reading comprehension system performing multi-hop reasoning over Context Graph facts.\n"
                "Each fact is '[id] lemma (sense): ROLE[role meaning]=Entity; ...'. Use the role meanings to read the "
                "direction of each relation (e.g. parent.01 ARG0 is the parent, ARG1 the child).\n"
                "RULES:\n"
                "1. Use only the facts provided; chain facts through shared entities.\n"
                "2. Containment is transitive: if A is located in B and B is located in C, A is in C. "
                "alias.01 links two names of the same entity.\n"
            )
            answer_form = "<exact entity name as written in the facts>"
        elif evidence_kind == "text":
            intro = (
                "You are a reading comprehension system performing multi-hop reasoning over text passages.\n"
                "RULES:\n"
                "1. Use only the passages provided; chain statements across passages through shared entities.\n"
                "2. Containment is transitive: if A is located in B and B is located in C, A is in C.\n"
            )
            answer_form = "<exact entity name as written in the passages>"
        else:
            intro = (
                "You are a question answering system.\n"
                "RULES:\n"
                "1. Answer the multi-hop question step by step.\n"
                "2. Give the shortest name that answers the question.\n"
            )
            answer_form = "<entity name>"
        # The refusal criterion is stated exactly once (rule 4) so policies cannot conflict.
        return (
            intro
            + "3. Reason concisely in 1-3 sentences.\n"
            + self._policy_rule(evidence_kind)
            + "5. End with exactly one final line, either\n"
            f"FINAL ANSWER: {answer_form}\n"
            "or\n"
            "FINAL ANSWER: NOT_IN_EVIDENCE"
        )

    def answer_question(
        self,
        question: str,
        blueprint: str,
        candidate_entities: Optional[List[str]] = None,
        evidence_kind: str = "graph",
    ) -> str:
        """Answers from Context Graph facts (evidence_kind="graph"), text passages ("text"), or
        without evidence ("none", closed-book). All variants share one output format and parser,
        so baselines and the graph system are scored identically."""
        candidates_clause = ""
        if candidate_entities:
            candidates_formatted = "\n".join(f"- {c}" for c in candidate_entities[:10])
            candidates_clause = f"\nENTITIES REACHABLE FROM THE QUESTION ENTITY:\n{candidates_formatted}\n"

        system_prompt = self._system_prompt(evidence_kind)
        if evidence_kind == "graph":
            user_prompt = (f"Context Graph Facts:\n{blueprint}\n{candidates_clause}\n"
                           f"Question: {question}\n\nAnswer the question. End with 'FINAL ANSWER: ...'.")
        elif evidence_kind == "text":
            user_prompt = (f"Passages:\n{blueprint}\n\n"
                           f"Question: {question}\n\nAnswer the question. End with 'FINAL ANSWER: ...'.")
        else:
            user_prompt = f"Question: {question}\n\nAnswer the question. End with 'FINAL ANSWER: ...'."
        try:
            response = self._chat(
                "answer",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=max(self.output_reserve_tokens, 1024),
                timeout=self.answer_timeout,
                extra_body=self.extra_body,
            )
            full_text = _extract_message_text(response.choices[0].message)
            self.last_answer_text = full_text
        except Exception as e:
            self.last_answer_text = f"<error: {e}>"
            print(f"\n[CRITICAL LLM CALL ERROR in answer_question]: {e}")
            return "STATUS: NOT_IN_EVIDENCE"

        matches = list(ANSWER_PATTERN.finditer(full_text))
        if not matches and full_text.strip():
            # The generation ended without a final line (usually truncated by the token limit).
            # A missing line is not a refusal: ask once, briefly, for the conclusion of the
            # reasoning already produced.
            full_text = self._finalize_truncated(system_prompt, user_prompt, full_text)
            matches = list(ANSWER_PATTERN.finditer(full_text))
        if matches:
            answer = self._clean_answer(matches[-1].group("answer"), question)
            return f"ANSWER: {answer}" if answer else "STATUS: NOT_IN_EVIDENCE"
        return "STATUS: NOT_IN_EVIDENCE"

    def _finalize_truncated(self, system_prompt: str, user_prompt: str, partial: str) -> str:
        """Continuation call that only asks for the final answer line."""
        tail = partial[-3000:]
        follow_up = (
            "Your reasoning was cut off. Based on it, output ONLY the final line now, either\n"
            "FINAL ANSWER: <exact entity name>\nor\nFINAL ANSWER: NOT_IN_EVIDENCE"
        )
        extra = dict(self.extra_body)
        extra["think"] = False
        extra["reasoning_effort"] = "none"
        try:
            response = self._chat(
                "finalize",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": tail},
                    {"role": "user", "content": follow_up},
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=self.answer_timeout,
                extra_body=extra,
            )
            final = _extract_message_text(response.choices[0].message)
        except Exception as e:
            final = f"<finalize error: {e}>"
        self.last_answer_text = f"{partial}\n[finalize] {final}"
        return final
