import numpy as np
import pandas as pd
from src import features


def test_mcq_flag():
    q = pd.Series([
        "What is 2+2?",
        "Pick one.\nAnswer Choices:\nA. 1\nB. 2",
        "Choose:\nA) red\nB) blue\nC) green",
    ])
    flags = features.mcq_flag(q).to_numpy()
    assert flags.tolist() == [False, True, True]


def test_handcrafted_shape_and_no_nan():
    q = pd.Series(["short", "find x " * 50, "Answer Choices: A. a B. b"])
    f = features.handcrafted_features(q)
    assert len(f) == 3
    assert not f.isna().any().any()
    assert "is_mcq" in f.columns and "log_char_len" in f.columns


def test_handcrafted_is_numeric_matrix():
    q = pd.Series(["a", "b b b"])
    arr = features.handcrafted_matrix(q)
    assert arr.dtype == np.float32
    assert arr.shape[0] == 2
