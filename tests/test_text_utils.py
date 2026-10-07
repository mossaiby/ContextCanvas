"""Unit tests for dependency-free normalisation, anchoring and parsing utilities."""
from semantics.text_utils import (
    chunk_text,
    entity_match_key,
    is_date_like,
    match_question_entities,
    normalize_surface,
    parse_bare_kin_noun,
    parse_kinship_phrase,
    slugify,
    split_place,
)


def test_entity_keys_merge_legal_suffixes_and_honorifics_only():
    assert entity_match_key("Bombardier Inc.") == entity_match_key("Bombardier Inc")
    assert entity_match_key("Rabat Ajax F.C.") == entity_match_key("Rabat Ajax")
    assert entity_match_key("Sir Charles Tupper") == entity_match_key("Charles Tupper")
    assert entity_match_key("Municipality of Nuevo Laredo") == entity_match_key("Nuevo Laredo Municipality")
    assert entity_match_key("King Crimson") != entity_match_key("Crimson")
    assert entity_match_key("Grant Green") != entity_match_key("Green")


def test_normalize_surface_preserves_relational_and_proper_names():
    assert normalize_surface("guitarist Grant Green") == "Grant Green"
    assert normalize_surface("Producer Records") == "Producer Records"
    assert normalize_surface("son of Peter Andreas Heiberg") == "son of Peter Andreas Heiberg"
    assert normalize_surface("Bombardier Inc.") == "Bombardier Inc."
    assert normalize_surface("Canada.") == "Canada"


def test_slugs_never_collapse_non_latin_names():
    assert slugify("北京") != slugify("東京")
    assert slugify("北京")


def test_date_detection():
    for v in ["May 4, 1948", "9 February 1953", "2010–11 season", "1914", "218 BC"]:
        assert is_date_like(v), v
    for v in ["3,721", "1st Baron Revelstoke", "Easton, Massachusetts", "Apollo 11"]:
        assert not is_date_like(v), v


def test_split_place_only_accepts_hierarchical_places():
    assert split_place("Canyon, Texas, United States") == ["Canyon", "Texas", "United States"]
    assert split_place("161,000 customers") is None
    assert split_place("6411 Orchard Avenue, Suite 101, Takoma Park") is None
    assert split_place("Philippe, Duke of Orléans") is None
    assert split_place("Calvert, Charles, and St. Mary's counties") is None


def test_kinship_parsing():
    assert parse_kinship_phrase("son of Peter Andreas Heiberg") == ("child", ["Peter Andreas Heiberg"])
    assert parse_kinship_phrase("elder sister of Lorenzo de' Medici") == ("sibling", ["Lorenzo de' Medici"])
    assert parse_kinship_phrase("wife of Bernardo Rucellai") == ("spouse", ["Bernardo Rucellai"])
    rel, targets = parse_kinship_phrase("born to Martin Luther and Katharina von Bora")
    assert rel == "child" and targets == ["Martin Luther", "Katharina von Bora"]
    assert parse_kinship_phrase("member of the band") is None
    assert parse_bare_kin_noun("younger son") == "child"
    assert parse_bare_kin_noun("President") is None


def test_question_anchoring_prefers_named_spans():
    names = [
        "employer", "Ulrich Walter", "Caroline LeRoy Webster", "Caroline", "Leroy", "Julia's House",
        "Madhepura district", "Unwinding: An Inner History of the New America", "author",
        "Bruce Lee Band", "Bruce Lee", "European Movement Germany", "Germany", "capital", "Pine Springs",
    ]
    stop = ["located"]
    assert match_question_entities("Where is Ulrich Walter's employer headquartered?", names, stop) == ["Ulrich Walter"]
    assert match_question_entities("Who is the child of Caroline LeRoy's spouse?", names, stop) == ["Caroline LeRoy Webster"]
    assert match_question_entities("What district is the headquarter of Julia's House located?", names, stop) == ["Julia's House"]
    assert match_question_entities("The Unwinding author volunteered for which organisation?", names, stop) == [
        "Unwinding: An Inner History of the New America"]
    assert match_question_entities("What label did the person in The Bruce Lee Band start?", names, stop) == ["Bruce Lee Band"]
    assert match_question_entities("What is the goal of the group European Movement Germany is in?", names, stop) == [
        "European Movement Germany"]
    assert match_question_entities("Where is Unregistered Lab?", names, stop) == []
    # Generic lower-case matches are only used when nothing name-like matches.
    assert match_question_entities("who was the author?", names, stop) == ["author"]


def test_chunk_text_respects_sentence_boundaries():
    text = "First sentence here. " * 60
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_acronym_alias_mining_and_possessives():
    from semantics.text_utils import is_placeholder, mine_acronym_aliases, parse_possessive_association
    text = ("He trained at the German Aerospace Center (DLR). He lived in London (UK). "
            "The Fair Labor Standards Act (FLSA) applies.")
    # Only abbreviations that spell the name's initials are mined (DLR is German; UK is a qualifier).
    assert mine_acronym_aliases(text) == [("Fair Labor Standards Act", "FLSA")]
    assert parse_possessive_association("DLR's Lander Control Center") == ("DLR", "Lander Control Center")
    assert parse_possessive_association("Swift's frustrations") is None
    assert is_placeholder("unknown") and is_placeholder("the unknown") and is_placeholder("it")
    assert not is_placeholder("Unknown Pleasures")


def test_alias_long_name_is_trimmed_to_the_abbreviation():
    from semantics.text_utils import mine_acronym_aliases
    text = "It was listed in High Point, North Carolina on the National Register of Historic Places (NRHP) in 1988."
    assert mine_acronym_aliases(text) == [("National Register of Historic Places", "NRHP")]


def test_alias_mining_rejects_tables_qualifiers_and_sentence_spans():
    from semantics.text_utils import mine_acronym_aliases
    assert mine_acronym_aliases("IPTV AT&T U-verse Channel 1652 HD Channel 652 (SD)") == []
    assert mine_acronym_aliases("The General Assembly of Georgia (USA) passed it.") == []
    assert mine_acronym_aliases("Worst Forms Convention (C182)") == []
    assert mine_acronym_aliases("Founded 1940. The Kolkata Knight Riders (KKR) play.") == [("Kolkata Knight Riders", "KKR")]


def test_names_with_middle_parts_and_plural_kinship():
    from semantics.text_utils import match_question_entities, parse_bare_kin_noun, parse_kinship_phrase
    names = ["Frances Amélia Tupper", "Anselm Tupper", "James Tupper", "Philippe I, Duke of Orléans", "Duke of Orléans"]
    assert match_question_entities("Where was the spouse of Frances Tupper born?", names) == ["Frances Amélia Tupper"]
    assert match_question_entities("Who is the grandmother of Philippe, Duke of Orléans?", names) == ["Philippe I, Duke of Orléans"]
    assert parse_kinship_phrase("children of Frances Tupper and Charles Tupper") is None
    assert parse_bare_kin_noun("sons") is None