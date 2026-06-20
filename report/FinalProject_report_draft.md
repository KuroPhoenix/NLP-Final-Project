# INLP 2026 Final Project — LLM Routing
### Report (Team 4) 


---

## 0. Problem analysis (design foundation)

The task is to route each query to one of 11 anonymized LLMs to maximize
**Reward₀.₈₅ = 0.85·mean(performance) − 0.15·mean(cost)/C̄ₘₐₓ**, where `C̄ₘₐₓ` is the mean per-row
maximum cost (≈ 0.0772 on train). Before modelling we profiled the data, and four facts shaped
every decision:

1. **The metric is a correctness problem with a cheap-model tie-break.** The oracle (per-row best
   choice) scores **0.673**; an "accuracy-oracle" that always picks the *cheapest correct* model
   scores **0.672**; and replacing each model's per-query cost with a *constant* per-model average
   still scores **0.672**. All three being equal means **per-query cost modelling is worthless** —
   the optimal policy is simply *"the cheapest model likely to be correct."*
2. **`performance` is ordinal `{0, 0.5, 1}`** (not binary) and the metric is linear in mean
   performance, so the natural target is *expected* performance / utility (a regression), not a class.
3. **Difficulty is bimodal:** 17.5% of queries are solved by all 11 models, 19.1% by none. On both
   extremes the optimal route is "a cheap model," so model identity only matters in the middle band.
4. **`Model_K` is the natural default:** near-free (cost ≈ 0.001) and correct 53% of the time — the
   highest reward of any fixed model (0.450, the *simple baseline*). The oracle routes ~51% of
   queries to it. Beating the baseline means finding the queries where K is wrong **and** another
   model is reliably right (only ~2,750 train rows carry the entire baseline→oracle gap).

We also confirmed **no train/test distribution shift** (an adversarial train-vs-test classifier
scores AUC ≈ 0.49, i.e. indistinguishable), so ordinary random K-fold CV is valid.

---

## Q1. How the router is implemented

Our submitted router is the **`router_a100`** pipeline: an *offline supervised router* that, for each
candidate model, regresses its **cost-aware utility** `Yᵢ = 0.85·perfᵢ − 0.15·costᵢ/C̄ₘₐₓ` directly,
then routes to `argmaxᵢ Ŷᵢ` with a **Model_K margin fallback**. Regressing the utility *directly*
(rather than predicting performance and recombining) is deliberate and central to why it works (Q2/Q3).

**Packages.** `scikit-learn` (Ridge, MultiOutputRegressor, TruncatedSVD, StandardScaler, KFold),
`LightGBM`, `sentence-transformers` (Qwen3-Embedding-4B), `numpy`/`pandas`/`scipy`; plus
`transformers` (ModernBERT-large; Qwen2.5-Instruct) for the Q3 experiments.

**Features / representations.**
- **TF-IDF**: word (1–2 gram, ≤120k) + char-wb (3–5 gram, ≤120k), sublinear TF, on text head/tail-
  capped to 60k chars.
- **Qwen3-Embedding-4B**: 1024-dim, mean-pooled over up to 3 chunks of 12k chars, L2-normalized.
- **Numeric query features**: char/word/line counts and digit/LaTeX/code/math-keyword/question-mark
  counts (+ logs), standardized and concatenated with the embeddings. These encode query *type*
  (MCQ / code / long-doc), which is strongly predictive of difficulty — MCQ queries average only
  ~0.14 model accuracy vs ~0.47 overall.

**Base learners** — each a multi-output regressor on the 11-dim utility target, trained with 5-fold
CV (KFold, shuffle, seed 42) to produce out-of-fold + full-data test predictions:

| Learner | Input | Model / loss |
|---|---|---|
| `tfidf_ridge` | TF-IDF | Ridge, L2 (`alpha=8`) |
| `qwen3_embedding_ridge` | emb ⊕ numeric | Ridge, L2 (`alpha=4`) |
| `qwen3_embedding_lgbm` | emb ⊕ numeric | LightGBM regression (900 trees, `lr=0.025`, `leaves=31`) |
| `qwen3_embedding_lgbm_two_head` | emb ⊕ numeric | two LightGBMs (perf-head + cost-head) recombined into utility |

**Ensembling & decision.** Predictions are combined two ways — a **weighted blend** whose convex
weights are grid-searched (step 0.1) to maximize OOF route reward, and a **rank blend**. The weighted
blend won, selecting **0.7 / 0.2 / 0.1** over `tfidf_ridge / qwen3_embedding_ridge /
qwen3_embedding_lgbm_two_head` (the plain embedding-LGBM was dropped). The final policy routes to
`argmax(utility)` plus the **Model_K margin-fallback** (Q2); among {best single learner, weighted
ensemble, ensemble+fallback} the highest-OOF-reward option is submitted. We later strengthened the
fallback to `q05` (Q2, the submitted best) and separately verified a **cross-fold-stable**
4-component refinement (`0.65/0.25/0.05/0.05`, adding embedding-LGBM). ‹Confirm which variant is your
chosen finalist.›

**Loss functions.** All learners minimize **L2** on the utility target (Ridge closed-form; LightGBM
`objective=regression`); the two-head learner minimizes L2 separately on `perf` and `cost` and
recombines `0.85·perf − 0.15·cost/C̄ₘₐₓ`.

**Validation methodology.** A single fixed 5-fold split shared by all learners; every decision judged
by **out-of-fold exact reward**, with a stricter **cross-fit** estimate (weights / calibration /
fallback margin re-selected without each evaluated fold) used to decide submissions. With only 3
submissions/day, we trusted CV over leaderboard probing.

**Key hyperparameters.** Ridge `alpha=8` (TF-IDF) / `alpha=4` (embeddings); LightGBM 900 trees,
`lr=0.025`, `num_leaves=31`, `min_child=30`; Qwen3-Embedding-4B dim 1024 (≤3×12k-char chunks);
ensemble weight-grid step 0.1; Model_K fallback margin = 5th-percentile OOF top-2 margin (`q05`).
The Q3 experiments add ModernBERT-large (`max_len` 2048, `lr=2e-5`, 3 epochs, bf16) and
Qwen2.5-3B-Instruct self-consistency (`k=4`).

---

## Q2. Balancing performance and cost (design intuition & motivation)

The metric weights performance heavily (0.85), but the cost term is *not* negligible: with
`C̄ₘₐₓ ≈ 0.077`, an expensive model (cost ≈ 0.04–0.07) is penalized ≈ 0.08–0.13, which can erase the
gain of being correct. From §0, the optimal balance is structurally simple: **pick the cheapest model
likely to be correct, and when in doubt default to the cheap-and-decent Model_K.** We realize this
with two deliberate levers:

1. **Regress the cost-aware utility directly.** Training on `Yᵢ = 0.85·perfᵢ − 0.15·costᵢ/C̄ₘₐₓ`
   *bakes the cost penalty into the target*: an expensive-but-capable model earns a high predicted
   utility only when it is confidently worth its cost. This proved critical. An alternative design
   that predicted *performance* and recombined `0.85·p̂ − cost` suffered a **winner's curse** —
   `argmax` over noisy per-model performance systematically routes to whichever capable-but-expensive
   model is *most over-predicted*, so the cost penalty lands on the hidden test split (that design
   scored 0.479 on CV but only **0.449** on the LB; Q3). Direct utility regression is robust to this.

2. **The Model_K confidence-margin fallback is the explicit knob.** When the top-2 predicted utilities
   are nearly tied, the router has no confident reason to leave the cheap default, so paying for a
   non-K model is a bad bet — route to K. The margin `τ` slides the trade-off: `τ=0` is aggressive
   (always chase the argmax, pay for performance), large `τ` is conservative (default to cheap K). At
   the 5th percentile it diverts only the least-confident ~5% of test queries to K and **improved the
   public LB from 0.46853 to 0.47007** for negligible CV cost — a pure robustness gain on the queries
   where confidence doesn't justify the spend.

Intuitively, the router **spends money only where it is confident the spend buys correctness**, and
defaults to free, reliable Model_K everywhere else — mirroring the oracle, which itself sends a
majority of queries to K.

---

## Q3. Methods compared, and which performed best

All rewards are exact Reward₀.₈₅. "CV" is 5-fold out-of-fold (cross-fit where noted); "LB" is the
public leaderboard. ‹verify LB values against Kaggle›.

| # | Method | CV reward | Public LB |
|---|---|---|---|
| 0 | **Simple baseline** — always Model_K | 0.4504 | ≈0.450 |
| 1 | **`router_a100`** — initial utility-regression ensemble (earlier run) | — | 0.45648 |
| 2 | **`router_a100` (exact-metric)** — 4-learner ensemble, weighted blend, `argmax` | ≈0.478 | 0.46853 |
| 3 | **(2) + Model_K `q05` margin fallback** — *submitted best* | 0.478 | **0.47007** |
| 4 | Correctness-first: predict perf → calibrate → `argmax` exp. reward + 11 per-model bias offsets | 0.4795 (in-sample) | 0.44936 |
| 5 | (4) made honest: nested calibration + cross-fit + always-K floor | 0.4631 (cross-fit) | 0.44590 |
| 6 | Combined: ensemble (3) + ModernBERT-large utility encoder + LLM-difficulty GBM, cross-fit gated | 0.4776 (= ensemble alone) | 0.46853 (collapses to base ensemble) |
| — | Oracle (upper bound) | 0.6731 | — |

**Best method: #3 — the `router_a100` cost-aware utility-regression ensemble with the Model_K
confidence fallback (public LB 0.47007).** Why it won:

- **Right target.** Regressing utility directly avoids the winner's-curse failure of the
  performance-then-route designs (#4, #5), whose impressive in-sample CV (0.479) collapsed *below the
  baseline* on the LB (0.449 / 0.446). This was our single most important lesson: with a hard
  `argmax` decision, optimizing the wrong (uncalibrated, over-tuned) objective overfits
  catastrophically.
- **Complementary representations.** TF-IDF (surface cues: is-MCQ, code, math, keywords) and Qwen3
  embeddings (semantics) capture the topic/difficulty structure that determines which model wins; the
  weighted blend keeps only the combination that maximizes OOF reward.
- **Robust decision rule.** The `q05` Model_K fallback is the only reliable gain we found beyond the
  raw ensemble (+0.0015 LB), by declining to pay for low-confidence routes; the lineage 0.45648 →
  0.46853 → 0.47007 shows each design refinement (exact-metric framing, then the fallback) paying off.
- **Honest evaluation prevented over-fitting the leaderboard.** After observing that in-sample tuning
  inflated CV by ~0.02–0.03 vs the LB, we adopted a cross-fit estimator (re-selecting
  weights/calibration/margin outside each evaluated fold). This discipline is *why* we did **not**
  ship the more complex method #6.

**Why the correctness-first reframe (#4, #5) failed.** It predicted per-model performance, calibrated
it, and routed by expected reward with tuned per-model bias offsets. Two compounding errors: (i) the
calibration, ensemble weights and 11 bias offsets were optimized on the *same* OOF that was reported
(in-sample inflation ~0.02–0.03); (ii) the winner's curse of Q2. Even after making the evaluation
honest (#5, cross-fit + always-K floor), its realistic value was only ≈ baseline — confirming the
approach, not just its tuning, was inferior to direct utility regression.

**Why the most complex method (#6) did *not* win.** We hypothesized that a fine-tuned ModernBERT-large
utility encoder and an **LLM-as-judge** difficulty signal (self-consistency entropy, solution length,
refusal rate, and a 1–10 difficulty rating from Qwen2.5-3B) would add orthogonal signal. Stacked onto
the validated ensemble under a strict cross-fit gate, **both new members received exactly zero weight
in all five folds**, and a fine-grained weight sweep confirmed a flat, noise-level response (peak
+0.0004, far below our +0.002 adoption threshold and within fold variance). Interpretation: the
validated ensemble already extracts the available topic/difficulty signal; with per-cell labels that
are single, noisy `{0,0.5,1}` observations, neither a fine-tuned encoder nor a 3B model's
self-consistency adds *transferable* routing information. We report this as a deliberately gated
**negative result** rather than chase a noise-level submission that risked regressing the private LB.
Submitted afterward as a sanity check, the combined file scored **0.46853** — exactly the base
ensemble (#2) and below the #3 fallback — empirically confirming on the leaderboard that the new
members contributed nothing.

---

## Appendix — reproduction

The original router is `FinalProject_router_a100.ipynb` (the `router_a100` pipeline above), which
writes per-learner OOF/test utility predictions and the weighted-ensemble submission. We additionally
re-implemented and extended the pipeline as a tested package under `src/` (`data`, `features`, `cv`,
`metric`, `models_classical`, `models_encoder`, `llm_difficulty`, `ensemble_routing`, `combine`,
`run`) with unit tests in `tests/`. The submitted best (#3) is the weighted ensemble's test
predictions routed by `argmax` with the Model_K `q05` margin fallback. ‹Add exact run commands.›
