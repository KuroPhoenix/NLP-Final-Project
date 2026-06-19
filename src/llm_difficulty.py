from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd

# Gold-free difficulty features from a general reasoning LLM: self-consistency over k samples,
# generation stats, and a 1-10 difficulty judgment. Parsing/aggregation helpers are pure and
# unit-tested; compute_llm_features drives vLLM (GPU) and caches the result.

FEATURE_COLS = [
    "sc_agreement", "sc_entropy", "sc_n_distinct",
    "gen_len_mean", "gen_len_std", "refuse_rate",
    "judge_difficulty", "judge_p_solvable",
]

_ANS_RE = re.compile(r"final answer\s*[:\-]?\s*(.+)", re.I)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MCQ_RE = re.compile(r"\b([A-E])\b")
_REFUSE_RE = re.compile(r"\b(cannot|can't|unable|not sure|don't know|impossible)\b", re.I)

SOLVE_PROMPT = ("Solve the problem. Show brief reasoning, then end with "
                "'Final answer: <answer>'.\n\nProblem:\n{q}")
JUDGE_PROMPT = ('Rate how hard this question is for a mid-tier language model. Respond with ONLY '
                'a JSON object: {{"difficulty": <integer 1-10>, "p_solvable": <float 0-1>}}.'
                '\n\nQuestion:\n{q}')


def extract_answer(text: str) -> str:
    """Best-effort final-answer extraction: \\boxed{}, 'Final answer:', MCQ letter, or last number."""
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        tail = m.group(1)
    else:
        m = _ANS_RE.search(text)
        if m:
            tail = m.group(1)
        else:
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            tail = lines[-1] if lines else ""
    tail = tail.strip()
    mcq = _MCQ_RE.findall(tail[:8])
    if mcq:
        return mcq[0].upper()
    nums = _NUM_RE.findall(tail)
    if nums:
        return nums[-1]
    return tail.lower()[:40]


def self_consistency_features(answers, lengths, refusals) -> dict:
    from collections import Counter
    n = max(len(answers), 1)
    c = Counter(a for a in answers if a != "")
    total = sum(c.values())
    top = c.most_common(1)[0][1] if c else 0
    probs = np.array([v / total for v in c.values()]) if total > 0 else np.array([1.0])
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "sc_agreement": top / n,
        "sc_entropy": entropy,
        "sc_n_distinct": float(len(c)),
        "gen_len_mean": float(np.mean(lengths)) if len(lengths) else 0.0,
        "gen_len_std": float(np.std(lengths)) if len(lengths) else 0.0,
        "refuse_rate": float(np.mean(refusals)) if len(refusals) else 0.0,
    }


def parse_judge(text: str):
    """Parse {'difficulty':1-10,'p_solvable':0-1} from judge output; clamp; default to (5, 0.5)."""
    diff, psolv = 5.0, 0.5
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            j = json.loads(m.group(0))
            diff = float(j.get("difficulty", 5))
            psolv = float(j.get("p_solvable", 0.5))
    except Exception:
        return 5.0, 0.5
    return float(np.clip(diff, 1, 10)), float(np.clip(psolv, 0, 1))


def _trim(q: str, max_chars: int = 6000) -> str:
    q = q or ""
    return q if len(q) <= max_chars else q[:4000] + "\n[...]\n" + q[-1500:]


def compute_llm_features(cfg, df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Drive vLLM to produce difficulty features for each query (GPU). Cached to parquet."""
    cache = cfg.cache_dir / f"llm_feats_{split}_{len(df)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from vllm import LLM, SamplingParams
    llm = LLM(model=cfg.llm_id, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    queries = [_trim(q) for q in df["query"].fillna("").astype(str).tolist()]
    sc_params = SamplingParams(n=cfg.llm_k_samples, temperature=cfg.llm_temperature,
                               top_p=0.95, max_tokens=cfg.llm_max_new_tokens)
    judge_params = SamplingParams(n=1, temperature=0.0, max_tokens=64)
    sc_out = llm.generate([SOLVE_PROMPT.format(q=q) for q in queries], sc_params)
    judge_out = llm.generate([JUDGE_PROMPT.format(q=q) for q in queries], judge_params)

    rows = []
    for s, j in zip(sc_out, judge_out):
        answers, lengths, refusals = [], [], []
        for o in s.outputs:
            t = o.text
            answers.append(extract_answer(t))
            lengths.append(len(t.split()))
            refusals.append(1 if _REFUSE_RE.search(t) else 0)
        feat = self_consistency_features(answers, lengths, refusals)
        d, p = parse_judge(j.outputs[0].text)
        feat["judge_difficulty"] = d
        feat["judge_p_solvable"] = p
        rows.append(feat)
    out = pd.DataFrame(rows)[FEATURE_COLS].astype(np.float32)
    out.to_parquet(cache)
    return out
