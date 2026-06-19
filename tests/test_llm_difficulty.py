import numpy as np
from src import llm_difficulty as ld


def test_extract_answer_boxed_mcq_numeric_plain():
    assert ld.extract_answer(r"work... \boxed{42}") == "42"
    assert ld.extract_answer("blah\nFinal answer: B") == "B"
    assert ld.extract_answer("the result is 1234") == "1234"
    assert ld.extract_answer("Final answer: -3.5") == "-3.5"
    assert ld.extract_answer("") == ""


def test_self_consistency_features_agreement_entropy():
    # 4 of 5 agree on "42"
    f = ld.self_consistency_features(["42", "42", "42", "42", "7"], [10, 12, 11, 9, 30], [0, 0, 0, 0, 1])
    assert abs(f["sc_agreement"] - 0.8) < 1e-9
    assert f["sc_n_distinct"] == 2.0
    assert f["sc_entropy"] > 0.0
    assert abs(f["refuse_rate"] - 0.2) < 1e-9
    # unanimous => zero entropy, agreement 1
    g = ld.self_consistency_features(["5", "5", "5"], [3, 3, 3], [0, 0, 0])
    assert abs(g["sc_agreement"] - 1.0) < 1e-9
    assert g["sc_entropy"] < 1e-9


def test_parse_judge_json_and_fallback():
    d, p = ld.parse_judge('Sure. {"difficulty": 9, "p_solvable": 0.1}')
    assert d == 9.0 and abs(p - 0.1) < 1e-9
    d2, p2 = ld.parse_judge("no json here")
    assert d2 == 5.0 and p2 == 0.5  # defaults
    d3, p3 = ld.parse_judge('{"difficulty": 99, "p_solvable": 2}')
    assert d3 == 10.0 and p3 == 1.0  # clamped


def test_feature_cols_constant():
    assert ld.FEATURE_COLS[0] == "sc_agreement"
    assert "judge_difficulty" in ld.FEATURE_COLS and "judge_p_solvable" in ld.FEATURE_COLS
