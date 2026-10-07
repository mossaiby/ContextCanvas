"""Tests for multi-model configuration, PropBank pinning and pilot sampling."""
import io
import json
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _cfg(tmp_path, **extra):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"base_url": "http://x/v1", "api_key": "k", **extra}))
    return str(p)


def _response(content):
    choice = MagicMock()
    choice.message.content = content
    return MagicMock(choices=[choice], usage=MagicMock(prompt_tokens=1, completion_tokens=1))


def test_extraction_and_answer_calls_use_their_own_models(tmp_path):
    from engine.realizer import Realizer
    r = Realizer(config_path=_cfg(tmp_path, extraction_model="small:9b", answer_model="big:27b"))
    r.client.chat.completions.create = MagicMock(return_value=_response('{"events": []}'))
    r.extract_structured_context("A owns B.")
    assert r.client.chat.completions.create.call_args.kwargs["model"] == "small:9b"
    r.client.chat.completions.create = MagicMock(return_value=_response("FINAL ANSWER: B"))
    r.answer_question("Who?", "facts")
    assert r.client.chat.completions.create.call_args.kwargs["model"] == "big:27b"


def test_model_name_is_default_and_missing_models_fail(tmp_path):
    from engine.realizer import Realizer
    r = Realizer(config_path=_cfg(tmp_path, model_name="m"))
    assert r.extraction_model == r.answer_model == "m"
    with pytest.raises(ValueError):
        Realizer(config_path=_cfg(tmp_path, answer_model="only-reader"))


def test_extraction_cache_is_keyed_by_extraction_model_only(tmp_path):
    from engine.realizer import Realizer
    cache = {"extraction_cache_dir": str(tmp_path / "cache")}
    a = Realizer(config_path=_cfg(tmp_path, extraction_model="e", answer_model="r1"), overrides=cache)
    b = Realizer(config_path=_cfg(tmp_path, extraction_model="e", answer_model="r2"), overrides=cache)
    c = Realizer(config_path=_cfg(tmp_path, extraction_model="e2", answer_model="r1"), overrides=cache)
    prompt = a._extraction_prompt()
    assert a._cache_path(prompt, "t") == b._cache_path(prompt, "t")   # readers share extractions
    assert a._cache_path(prompt, "t") != c._cache_path(prompt, "t")   # extractors do not


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


FRAME = b'<frameset><predicate lemma="own"><roleset id="own.01" name="possess"><roles><role n="0" descr="owner"/></roles></roleset></predicate></frameset>'


def _fake_github(archive, known_ref="v3.4.0", sha="a" * 40):
    def fetch(url):
        if "/commits/" in url:
            if url.endswith("/" + known_ref) or url.endswith("/" + sha):
                return json.dumps({"sha": sha}).encode()
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        if "/tags" in url:
            return json.dumps([{"name": known_ref}]).encode()
        if "codeload" in url:
            assert url.endswith(sha), "must download the resolved commit, not a moving branch"
            return archive
        raise AssertionError(url)
    return fetch


def test_download_pins_commit_and_verifies_checksum(tmp_path):
    from download_propbank import download
    archive = _zip({"propbank-frames-aaa/frames/own.xml": FRAME, "propbank-frames-aaa/README.md": b"x"})
    paths = dict(lock_path=tmp_path / "lock.json", frames_dir=tmp_path / "frames", cache_path=tmp_path / "cache.json")
    lock = download("v3.4.0", update=False, fetch=_fake_github(archive), **paths)
    assert lock["commit"] == "a" * 40 and lock["frame_files"] == 1 and (tmp_path / "frames" / "own.xml").exists()
    # A later run reuses the locked commit even without --ref ...
    assert download(None, update=False, fetch=_fake_github(archive), **paths)["commit"] == "a" * 40
    # ... and refuses a changed archive for the same commit.
    tampered = _zip({"propbank-frames-aaa/frames/own.xml": FRAME + b" "})
    with pytest.raises(SystemExit):
        download(None, update=False, fetch=_fake_github(tampered), **paths)
    # Asking for another release requires --update.
    with pytest.raises(SystemExit):
        download("v9.9", update=False, fetch=_fake_github(archive), **paths)


def test_unknown_ref_lists_available_tags(tmp_path):
    from download_propbank import download
    try:
        download("v3.4", update=True, fetch=_fake_github(_zip({})), lock_path=tmp_path / "l.json",
                 frames_dir=tmp_path / "f", cache_path=tmp_path / "c.json")
    except SystemExit as exc:
        assert "v3.4.0" in str(exc)
    else:
        raise AssertionError("an unknown ref must stop the download")
    assert not (tmp_path / "l.json").exists()


def test_catalog_rebuilds_when_the_locked_release_changes(tmp_path):
    from semantics.propbank import PropBankCatalog
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "own.xml").write_bytes(FRAME)
    (tmp_path / "propbank_lock.json").write_text(json.dumps({"ref": "v3.4.0", "commit": "c1"}))
    cat = PropBankCatalog(frames_dir=frames, cache_file=tmp_path / "propbank_cache.json")
    assert cat.commit == "c1"
    assert json.loads((tmp_path / "propbank_cache.json").read_text())["_meta"]["commit"] == "c1"
    (tmp_path / "propbank_lock.json").write_text(json.dumps({"ref": "v3.5.0", "commit": "c2"}))
    PropBankCatalog(frames_dir=frames, cache_file=tmp_path / "propbank_cache.json")
    assert json.loads((tmp_path / "propbank_cache.json").read_text())["_meta"]["commit"] == "c2"


def test_pilot_pool_uses_only_development_questions():
    from evaluation.experiments import draw_sample
    recs = [{"id": f"q{i}", "answerable": True} for i in range(100)]
    pilot = draw_sample(recs, n=20, seed=1, exclude_first=50, pool="dev")
    test = draw_sample(recs, n=50, seed=1, exclude_first=50, pool="test")
    assert set(pilot["test_ids"]) <= set(pilot["dev_ids"])
    assert not set(test["test_ids"]) & set(test["dev_ids"])


def test_resume_drops_a_truncated_last_line(tmp_path):
    from evaluation.experiments import done_ids
    p = tmp_path / "predictions.jsonl"
    p.write_text('{"id": "q1"}\n{"id": "q2"}\n{"id": "q3", "pred')
    assert done_ids(p) == {"q1", "q2"}
    assert p.read_text() == '{"id": "q1"}\n{"id": "q2"}\n'


def test_model_comparison_table(tmp_path):
    from evaluation.report import build_models
    ids = [f"q{i}" for i in range(10)]
    dirs = []
    for label, reader, cc_correct in (("main", "qwen3.8:27b", 8), ("gemma", "gemma4:26b", 6)):
        d = tmp_path / label
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"test_ids": ids, "extraction_model": "qwen3.5:9b", "answer_model": reader}))
        for system, correct in (("contextcanvas", cc_correct), ("full_context", 5)):
            (d / system).mkdir()
            with open(d / system / "predictions.jsonl", "w") as f:
                for i, q in enumerate(ids):
                    ok = float(i < correct)
                    f.write(json.dumps({"id": q, "hops": 2, "answerable": True, "em": ok, "f1": ok, "refused": False,
                                        "usage": {}}) + "\n")
        dirs.append((label, d))
    (tmp_path / "gen").mkdir()
    build_models(dirs, tmp_path / "gen")
    table = (tmp_path / "gen" / "tab_models.tex").read_text()
    assert "gemma4:26b" in table and "80.0" in table and "+30.0" in table
