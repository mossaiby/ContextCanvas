"""LLM judge for the extraction-quality study, via any OpenAI-compatible chat API.

    python -m evaluation.llm_judge annotation/sample.csv --config configs/judge.json \
        --out annotation/annotator_llm.csv

Reads the exported annotation sample, asks the judge model to annotate each paragraph by the
rules in ANNOTATION_GUIDELINES.md (the text between the judge-rules markers, verbatim), and
writes a CSV in exactly the format the human annotators fill, so evaluation/annotation.py
scores the judge against the humans like any other annotator.

Outputs next to --out:
    <name>.csv          the judge's annotations
    <name>.log.jsonl    every request and raw response (resume point and audit trail)
    <name>.meta.json    model, endpoint, rules hash, settings, token usage

Reproducibility: the rules hash, model name and settings are recorded and the raw responses are
logged. Hosted models change or are retired, so the logged responses, not a re-run, are the
record of what the judge said. A run resumes from the log; responses made under different rules
or a different model are not reused.

Config (JSON):
    base_url      e.g. "https://api.openai.com/v1" (any OpenAI-compatible endpoint works)
    api_key_env   name of the environment variable holding the key, e.g. "OPENAI_API_KEY"
    model         the judge model; choose a strong one from a different family than the
                  extractor and readers
    temperature   0 for determinism; dropped automatically (and recorded) if the model rejects it
    max_tokens    output budget per paragraph
    concurrency   parallel requests
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.annotation import FIELDS  # noqa: E402

PLACEHOLDER_MODEL = "<set-your-judge-model>"
RULES_START, RULES_END = "<!-- judge-rules:start -->", "<!-- judge-rules:end -->"

OUTPUT_SPEC = """
## Your task

You are one of the annotators. You receive one paragraph and the events a system extracted
from it, numbered from 0. Apply the rules above exactly as a careful human annotator would.
Judge only what the paragraph states, not what you know about the world.

Return ONLY a JSON object, no other text:
{
  "events": [
    {"index": 0, "event_correct": 1, "roles_correct": 1, "context_correct": null, "reason": "<= 15 words"}
  ],
  "missed_facts": 0,
  "missed_examples": ["<one short line per missed relation>"]
}

* Include every event index exactly once.
* event_correct is 1 or 0.
* roles_correct is 1 or 0 when event_correct is 1, and null when event_correct is 0.
* context_correct is 1 or 0 only when event_correct is 1 AND the event shows a time or place;
  otherwise null.
* missed_facts is the count for the whole paragraph; missed_examples lists them (empty if 0).
""".strip()


# ------------------------------------------------------------------------------- inputs

def load_rules(guidelines: Path) -> str:
    text = guidelines.read_text(encoding="utf-8")
    if RULES_START not in text or RULES_END not in text:
        raise SystemExit(f"{guidelines} has no {RULES_START} ... {RULES_END} section.")
    return text.split(RULES_START, 1)[1].split(RULES_END, 1)[0].strip()


def group_paragraphs(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Rows of one paragraph are contiguous; the text is on the first row. Paragraphs with no
    extracted events have a single row with an empty lemma."""
    paragraphs: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not paragraphs or paragraphs[-1]["key"] != row["paragraph_key"]:
            paragraphs.append({"key": row["paragraph_key"], "text": row.get("paragraph_text", ""), "rows": []})
        paragraphs[-1]["rows"].append(i)
    return paragraphs


def _has_context(row: Dict[str, str]) -> bool:
    return bool(row.get("time", "").strip() or row.get("place", "").strip())


def build_messages(rules: str, paragraph: Dict[str, Any], rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    events = []
    for idx in paragraph["rows"]:
        r = rows[idx]
        if not r.get("lemma"):
            continue
        line = (f"[{r['event_index']}] {r['lemma']} ({r['sense_id']}): roles {r['roles']}"
                + (f"; role meanings {r['role_meanings']}" if r.get("role_meanings") else "")
                + (f"; time {r['time']}" if r.get("time") else "")
                + (f"; place {r['place']}" if r.get("place") else ""))
        events.append(line)
    user = (f"PARAGRAPH:\n{paragraph['text']}\n\nEXTRACTED EVENTS:\n"
            + ("\n".join(events) if events else "(none - only count missed_facts)"))
    return [{"role": "system", "content": rules + "\n\n" + OUTPUT_SPEC}, {"role": "user", "content": user}]


# ------------------------------------------------------------------------------- outputs

def parse_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the response")
    return json.loads(text[start:end + 1])


def _binary(value: Any, what: str) -> int:
    if value in (0, 1) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise ValueError(f"{what} must be 0 or 1, got {value!r}")


def validate(parsed: Dict[str, Any], paragraph: Dict[str, Any], rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Checks the judgment against the rules' structure and normalizes it. Conditional fields
    are blanked where the rules say they must be blank, so the judge cannot add judgments a
    human would not make."""
    event_rows = {int(rows[i]["event_index"]): rows[i] for i in paragraph["rows"] if rows[i].get("lemma")}
    given = parsed.get("events", [])
    if not isinstance(given, list):
        raise ValueError("'events' must be a list")
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in given:
        idx = int(item.get("index", -1))
        if idx not in event_rows:
            raise ValueError(f"unknown event index {idx}")
        if idx in by_index:
            raise ValueError(f"event index {idx} judged twice")
        ev = _binary(item.get("event_correct"), f"event {idx} event_correct")
        out = {"event_correct": ev, "roles_correct": None, "context_correct": None, "reason": str(item.get("reason", ""))[:200]}
        if ev == 1:
            out["roles_correct"] = _binary(item.get("roles_correct"), f"event {idx} roles_correct")
            if _has_context(event_rows[idx]):
                out["context_correct"] = _binary(item.get("context_correct"), f"event {idx} context_correct")
        by_index[idx] = out
    missing = sorted(set(event_rows) - set(by_index))
    if missing:
        raise ValueError(f"events not judged: {missing}")
    missed = parsed.get("missed_facts")
    if not isinstance(missed, int) or isinstance(missed, bool) or missed < 0:
        raise ValueError(f"missed_facts must be a non-negative integer, got {missed!r}")
    return {"events": by_index, "missed_facts": missed, "missed_examples": parsed.get("missed_examples", [])}


# ------------------------------------------------------------------------------- judge

class Judge:
    def __init__(self, cfg: Dict[str, Any], client: Any = None):
        self.cfg = cfg
        self.model = cfg["model"]
        if self.model == PLACEHOLDER_MODEL:
            raise SystemExit("Set 'model' in the judge config to the judge model you want to use.")
        self.temperature: Optional[float] = cfg.get("temperature", 0.0)
        self.use_json_mode = True
        self.token_param = "max_completion_tokens"  # OpenAI; older compatible servers use max_tokens
        self.dropped: List[str] = []
        self._lock = threading.Lock()
        if client is None:
            from openai import OpenAI
            key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
            if not key:
                raise SystemExit(f"Environment variable {cfg.get('api_key_env', 'OPENAI_API_KEY')} is not set.")
            client = OpenAI(base_url=cfg.get("base_url"), api_key=key, timeout=float(cfg.get("timeout_s", 300)),
                            max_retries=int(cfg.get("max_retries", 3)))
        self.client = client

    def _call(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        """One chat call. Parameters a model rejects (temperature on some reasoning models, JSON
        mode on some endpoints) are dropped once, for all later calls, and recorded in the meta."""
        for _ in range(4):
            kwargs: Dict[str, Any] = {"model": self.model, "messages": messages,
                                      self.token_param: int(self.cfg.get("max_tokens", 4000))}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # parameter not supported by this model/endpoint
                msg = str(exc).lower()
                with self._lock:
                    if self.token_param == "max_completion_tokens" and "max_completion_tokens" in msg:
                        self.token_param = "max_tokens"
                        self.dropped.append("max_completion_tokens->max_tokens")
                        continue
                    if self.temperature is not None and "temperature" in msg:
                        self.temperature = None
                        self.dropped.append("temperature")
                        continue
                    if self.use_json_mode and ("response_format" in msg or "json" in msg):
                        self.use_json_mode = False
                        self.dropped.append("response_format")
                        continue
                raise
            usage = getattr(resp, "usage", None)
            tokens = {"prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                      "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}
            return resp.choices[0].message.content or "", tokens
        raise RuntimeError("request failed after dropping unsupported parameters")

    def judge(self, messages: List[Dict[str, str]], paragraph: Dict[str, Any], rows: List[Dict[str, str]]) -> Dict[str, Any]:
        """Asks once; if the answer breaks the format, asks once more quoting the problem."""
        attempts = []
        convo = list(messages)
        for attempt in range(2):
            raw, tokens = self._call(convo)
            record = {"attempt": attempt, "raw": raw, "usage": tokens}
            try:
                record["result"] = validate(parse_json(raw), paragraph, rows)
                attempts.append(record)
                return {"ok": True, "attempts": attempts, "result": record["result"]}
            except (ValueError, json.JSONDecodeError) as exc:
                record["error"] = str(exc)
                attempts.append(record)
                convo = messages + [{"role": "assistant", "content": raw},
                                    {"role": "user", "content": f"Your answer was invalid: {exc}. Return the corrected JSON only."}]
        return {"ok": False, "attempts": attempts, "result": None}


def _result_from_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(entry["result"])
    res["events"] = {int(k): v for k, v in res["events"].items()}
    return res


def run(sample: Path, cfg: Dict[str, Any], out: Path, guidelines: Path, client: Any = None) -> Dict[str, Any]:
    with open(sample, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rules = load_rules(guidelines)
    rules_sha = hashlib.sha256(rules.encode("utf-8")).hexdigest()
    judge = Judge(cfg, client)
    paragraphs = group_paragraphs(rows)

    log_path, meta_path = out.with_suffix(".log.jsonl"), out.with_suffix(".meta.json")
    done: Dict[str, Dict[str, Any]] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("ok") and entry.get("model") == judge.model and entry.get("rules_sha256") == rules_sha:
                done[entry["paragraph_key"]] = _result_from_log(entry)
    todo = [p for p in paragraphs if p["key"] not in done]
    print(f"Judge {judge.model}: {len(done)} paragraphs reused from log, {len(todo)} to judge")

    out.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    failures: List[str] = []

    def work(p: Dict[str, Any]) -> None:
        verdict = judge.judge(build_messages(rules, p, rows), p, rows)
        entry = {"paragraph_key": p["key"], "model": judge.model, "rules_sha256": rules_sha, "ok": verdict["ok"],
                 "attempts": verdict["attempts"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if verdict["ok"]:
            entry["result"] = {**verdict["result"], "events": {str(k): v for k, v in verdict["result"]["events"].items()}}
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if verdict["ok"]:
                done[p["key"]] = verdict["result"]
            else:
                failures.append(p["key"])
            print(f"  {len(done) + len(failures)}/{len(paragraphs)} {p['key'][:12]} {'ok' if verdict['ok'] else 'FAILED'}")

    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 4))) as pool:
        list(pool.map(work, todo))

    # Write the annotation file in the humans' format; paragraphs that failed stay blank.
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in paragraphs:
            res = done.get(p["key"])
            for n, idx in enumerate(p["rows"]):
                row = {k: rows[idx].get(k, "") for k in FIELDS}
                if res:
                    ev = res["events"].get(int(row["event_index"])) if row.get("lemma") else None
                    if ev:
                        row["event_correct"] = str(ev["event_correct"])
                        row["roles_correct"] = "" if ev["roles_correct"] is None else str(ev["roles_correct"])
                        row["context_correct"] = "" if ev["context_correct"] is None else str(ev["context_correct"])
                    if n == 0:
                        row["missed_facts"] = str(res["missed_facts"])
                writer.writerow(row)

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                for a in json.loads(line).get("attempts", []):
                    for k in usage:
                        usage[k] += a.get("usage", {}).get(k, 0)
            except json.JSONDecodeError:
                pass
    meta = {
        "model": judge.model, "base_url": cfg.get("base_url"), "rules_sha256": rules_sha,
        "guidelines": str(guidelines), "temperature": judge.temperature, "json_mode": judge.use_json_mode,
        "dropped_parameters": sorted(set(judge.dropped)), "max_tokens": cfg.get("max_tokens", 4000),
        "paragraphs": len(paragraphs), "judged": len(done), "failed": sorted(failures),
        "usage_total": usage, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(done)}/{len(paragraphs)} paragraphs judged; {len(failures)} failed)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", type=Path, help="CSV written by `evaluation.annotation export`")
    ap.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "judge.json")
    ap.add_argument("--out", type=Path, default=Path("annotation/annotator_llm.csv"))
    ap.add_argument("--guidelines", type=Path, default=ROOT_DIR / "ANNOTATION_GUIDELINES.md")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.sample, cfg, args.out, args.guidelines)


if __name__ == "__main__":
    main()
