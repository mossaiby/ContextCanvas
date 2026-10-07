"""Edge cases of the string utilities (semantics/text_utils.py)."""
from semantics.text_utils import (
    _name_variants, is_date_like, match_question_entities, parse_kinship_phrase, slugify, split_place,
)


def test_slugify_never_collapses_and_bounds_length():
    assert slugify("!!!").startswith("h") and slugify("!!!") != slugify("???")
    long_a, long_b = slugify("x" * 80), slugify("x" * 81)
    assert len(long_a) <= 67 and long_a != long_b


def test_degenerate_dates_and_places():
    assert is_date_like("—") is False
    assert split_place("Canyon,") is None                                  # empty part
    assert split_place("Canyon, Big Long Name Here For You") is None       # part too long to be a place


def test_kinship_needs_a_proper_name():
    assert parse_kinship_phrase("son of a farmer") is None
    assert parse_kinship_phrase("son of Peter Heiberg") == ("child", ["Peter Heiberg"])


def test_name_variants_drop_disambiguators():
    assert _name_variants("Mercury (planet)") == ["Mercury (planet)", "Mercury"]


def test_question_matching_ignores_stopwords_and_function_words():
    assert match_question_entities("Who founded of?", ["of", "A", "Paris"], stopwords=["Paris"]) == []


def test_exact_match_beats_prefix_on_the_same_span():
    assert match_question_entities("Who was Caroline LeRoy?", ["Caroline LeRoy", "Caroline LeRoy Webster"]) == ["Caroline LeRoy"]


def test_prefix_matches_per_span_are_capped():
    names = ["Caroline LeRoy Webster", "Caroline LeRoy Smith", "Caroline LeRoy Jones", "Caroline LeRoy Brown"]
    assert match_question_entities("Who was Caroline LeRoy?", names) == names[:3]
