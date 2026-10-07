"""Edge cases of stats, baselines and the PropBank catalog."""
import json
import math

import pytest

from evaluation.baselines import run_text_baseline
from evaluation.stats import bootstrap_ci, cohens_kappa, paired_bootstrap_p, set_f1
from semantics.propbank import PropBankCatalog, _canonical_roleset_id


def test_stats_degenerate_inputs():
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    with pytest.raises(ValueError):
        paired_bootstrap_p([1.0], [1.0, 0.0])
    assert set_f1([], [1]) == 0.0 and set_f1([1], [2]) == 0.0
    assert math.isnan(cohens_kappa([1], [1, 0]))


class _Reader:
    def __init__(self):
        self.calls = []

    def answer_question(self, question, evidence, evidence_kind):
        self.calls.append(evidence)
        return "ANSWER: x"


def test_full_context_reads_every_paragraph_and_unknown_baselines_fail():
    paragraphs = [{"idx": i, "title": f"T{i}", "paragraph_text": "text"} for i in range(3)]
    out = run_text_baseline(_Reader(), {"question": "q", "paragraphs": paragraphs}, "full_context", 10_000)
    assert out["support"] == [0, 1, 2]
    with pytest.raises(ValueError):
        run_text_baseline(_Reader(), {"question": "q", "paragraphs": paragraphs}, "magic", 10_000)


def test_canonical_roleset_ids():
    assert _canonical_roleset_id("Abandon-1") == "abandon.01"
    assert _canonical_roleset_id("co-write.01") == "co-write.01"
    assert _canonical_roleset_id("nosense") == "nosense"


FRAMES = """<frameset>
 <predicate lemma="run">
  <roleset id="run.01" name="operate, proceed"><roles><role n="0" descr="operator"/><role n="1" descr="machine, operation"/></roles></roleset>
  <roleset id="run.02" name="walk quickly, a course or contest"><roles><role n="0" descr="runner"/><role n="1" descr="course, race"/></roles></roleset>
  <roleset id="broken" name="no sense number"/>
 </predicate>
 <predicate lemma="sing"><roleset id="sing.01" name="make music"><roles><role n="0" descr="singer"/><role n="M" descr="mod"/></roles></roleset></predicate>
</frameset>"""


@pytest.fixture
def catalog_dir(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "run.xml").write_text(FRAMES)
    (frames / "bad.xml").write_text("<frameset><unclosed>")
    return tmp_path


def test_catalog_parses_xml_skips_bad_files_and_disambiguates(catalog_dir):
    (catalog_dir / "propbank_lock.json").write_text("{not json")          # unreadable lock -> unpinned
    cat = PropBankCatalog(frames_dir=catalog_dir / "frames", cache_file=catalog_dir / "propbank_cache.json")
    assert cat.lock is None and cat.commit is None
    assert "run.01" in cat.framesets and "broken" not in cat.framesets
    assert cat.framesets["sing.01"].roles == {":ARG0": "singer"}           # non-numbered roles ignored
    assert cat.disambiguate_sense("sing", "anything").roleset_id == "sing.01"          # single sense
    assert cat.disambiguate_sense("run", "She won the race on the course").roleset_id == "run.02"
    assert cat.disambiguate_sense("unknownverb", "x") is None


def test_catalog_reads_its_own_cache_and_legacy_flat_caches(catalog_dir):
    cache = catalog_dir / "propbank_cache.json"
    PropBankCatalog(frames_dir=catalog_dir / "frames", cache_file=cache)
    assert "framesets" in json.loads(cache.read_text())
    reloaded = PropBankCatalog(frames_dir=catalog_dir / "nowhere", cache_file=cache)
    assert "run.02" in reloaded.framesets                                  # served from the cache
    cache.write_text(json.dumps({"Walk-01": {"name": "walk", "description": "walk", "roles": {}}}))
    legacy = PropBankCatalog(frames_dir=catalog_dir / "nowhere", cache_file=cache)
    assert "walk.01" in legacy.framesets


def test_catalog_without_frames_or_cache_still_has_system_frames(tmp_path):
    cat = PropBankCatalog(frames_dir=tmp_path / "none", cache_file=tmp_path / "cache.json")
    assert "parent.01" in cat.framesets and "run.01" not in cat.framesets
