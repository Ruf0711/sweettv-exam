# Stage 2 observed results

Completed on 2026-10-22 using Python 3.10.11. All original Stage 1 source,
configuration, tests, data and artifacts passed preservation checks (58 files).
The Stage 1 cumulative Popularity MAP@5 was reproduced exactly:
`0.7950661375661375`. No Kaggle upload, final-test prediction file or PDF was
generated.

## Evaluation and selection

Primary proxy: **strictly over 80% of one broadcast in one uninterrupted
same-channel session**. The choice follows the third-party account by
[Nikita Melentev, 2021-03-20](https://towardsdatascience.com/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156/),
not independently verified official evaluation code. Inclusive 50% completion
and the original cumulative 60-second proxy remain available.

Elapsed times use UTC after explicit Europe/Kyiv localization. Cutoffs clip
observed viewing, never the full-broadcast denominator. Repeated broadcasts
and separate sittings do not accumulate into completion. Conflicting broadcast
seconds are excluded, including conflicts with an unknown programme ID.
Channel changes truncate the old channel without automatically resuming it.

All scores below average over the same **1,260 template users**, with empty
relevance contributing zero. Every model uses the same candidates within each
fold and emits five distinct positive IDs in exact template order.

- Inner tuning: `[2020-04-20, 2020-05-18)` Kyiv; selected parameters were frozen
  before outer evaluation.
- Main holdout: `[2020-05-18, 2020-07-27)` Kyiv; 6,363 scheduled catalogue IDs.
- Recent holdout: `[2020-07-13, 2020-07-27)` Kyiv; 2,697 window-overlap IDs.
  The shorter horizon reduces accumulated-label saturation and tests recent
  behaviour. These two outer periods overlap and are not independent folds.

### Fixed-cohort MAP@5

| Model | Main >80% | Main >=50% | Main >=60s | Recent >80% | Recent >=50% | Recent >=60s |
|---|---:|---:|---:|---:|---:|---:|
| Global Popularity | 0.439197 | 0.529705 | **0.795066** | 0.250356 | 0.308376 | 0.557140 |
| Personal History | 0.509021 | 0.596215 | 0.734714 | **0.361846** | **0.429228** | 0.594043 |
| Sparse implicit ALS | 0.370304 | 0.462927 | 0.688331 | 0.219481 | 0.271346 | 0.495164 |
| Content-Based | 0.169928 | 0.200806 | 0.309577 | 0.107137 | 0.120752 | 0.183608 |
| History/ALS hybrid | **0.513351** | **0.602519** | 0.755724 | 0.356853 | 0.426114 | **0.609964** |

### Relevance coverage after candidate filtering

| Rule | Main evaluable users | Main user/programme pairs | Recent evaluable users | Recent user/programme pairs |
|---|---:|---:|---:|---:|
| Cumulative >=60s | 1,246 | 229,853 | 1,233 | 80,523 |
| Continuous >=50% | 1,244 | 51,842 | 1,211 | 18,992 |
| Continuous >80% | 1,235 | 31,811 | 1,175 | 12,218 |

The completion-based evaluation substantially changes the interpretation of
Stage 1's high popularity score. The strict primary proxy has 25 empty-label
users on the main fold and 85 on the recent fold; all remain in the denominator.

## Implemented methods and parameters

- **Global Popularity:** unchanged distinct-viewer counts for any positive
  training support, all training history; ascending-ID tie-breaking.
- **Personal History:** completed-broadcast (`>=50%`) frequency, duration and
  recency: `log1p(count) * (1 + 0.25*log1p(seconds/3600)) * 2**(-age_days/90)`.
  Previously viewed programmes are permitted; unsupported positions and cold
  users use global popularity. There are 15 completion-cold template users at
  the main cutoff and 6 at the recent cutoff.
- **ALS:** genuine compiled CPU sparse implicit-feedback ALS (`implicit==0.7.2`),
  seed 42, one thread, 20 iterations. The eight-point inner grid was factors
  `{32,64}` x regularization `{0.05,0.2}` x confidence scale `{5,20}`.
  Selected: **32 factors, regularization 0.05, scale 5**. Nonzero confidence is
  `1 + scale*log1p(history_weight)`; missing feedback has confidence 1. Main
  training has 159,948 nonzeros; recent training has 259,268. Cold users use
  popularity; cold items have no fabricated latent factors and receive the
  popularity fallback/tie prior.
- **Content-Based:** character TF-IDF `(3,5)` title n-grams, maximum 30,000 title
  features, title weight 0.4, sparse category/genre/channel/person/decade
  features, and normalized sparse user profiles with at most 1,024 features
  per user. Vocabulary, IDF and feature dictionaries fit only pre-cutoff
  metadata. Upcoming cold items use frozen transformations. It provides
  nonempty content features for 4,379 of 4,380 completion-cold main candidates
  and all 1,262 completion-cold recent candidates.
- **Hybrid:** reciprocal-rank fusion, constant 60, 75% History / 25% ALS.
  Four small History+ALS/Content trials were evaluated after individual models.
  Inner primary MAP@5 improved from History's `0.446960` to `0.457650`.

## Runtime and resources

Completion preparation took **63.04 s**, with a Windows peak working set of
**2.04 GiB**. The initial comparison including uncached window creation took
52.61 s. The final verified comparison with those caches reused took
**43.23 s**, including tuning, both holdouts and artifact writes, with a
**0.98 GiB** Windows peak working set. Outer predictions and scores were
bit-for-bit identical between the two comparison executions.

| Model | Main fit seconds | Main score seconds | Recent fit seconds | Recent score seconds |
|---|---:|---:|---:|---:|
| Global Popularity | 0.122 | 0.006 | 0.174 | 0.003 |
| Personal History | 0.024 | 0.028 | 0.035 | 0.015 |
| ALS | 0.883 | 0.039 | 1.284 | 0.027 |
| Content-Based | 5.118 | 0.382 | 4.732 | 0.230 |
| Hybrid, components included | 0.906 | 0.637 | 1.319 | 0.334 |

These fit/score times exclude label construction, evaluation and artifact writes.
Hybrid scoring includes fusion (0.570 s main; 0.291 s recent). RSS measurements
are for the shared process, not isolated per-model allocations.

## Verification

- **79 tests passed**, zero failures/errors/skips, 2.3925 s. This includes all
  original 50, exact threshold boundaries, switches, repeated broadcasts,
  unknown IDs, DST, cutoffs, leakage, sparse training, reproducibility, cold
  starts and Top-5 validity. In addition to Stage 1's 100 dense-reference
  cases, 80 randomized session and 80 randomized broadcast cases passed.
- The run checker verified 25 model/fold outputs, all **157,500 prediction
  positions**, 11 evidence/window caches and exact recomputation of all three
  metrics.
- The independent checker verified **3,553,650** broadcast/session events,
  **85,837** distinct broadcast instances against schedule metadata, every row
  of all six completion tables, **75** aggregate metric values and **94,500**
  per-user AP values with exact rational arithmetic. It also reconstructed
  the two example users' 3 and 6,577 continuous sessions directly from raw logs.
  This independent check took 23.99 s.

## Recommendation and limitations

**Recommend Personal History with Global Popularity fallback as the robust
default.** It improves primary MAP@5 over popularity by **0.069824** on the main
holdout and **0.111490** on the recent holdout. The inner-selected hybrid has
the highest main score but gains only 0.004330 over History there and loses
0.004993 on the recent primary proxy. Its incremental benefit is inconsistent.
ALS is stronger than Content-Based among the two required exam methods;
neither standalone method surpasses Personal History in these experiments.

Official hidden labels remain unverified. Raw logs lack device identifiers,
so concurrent-channel sessions require the documented conservative policy.
Future schedule metadata is assumed available for catalogue construction and
cold-item transformation, but no future viewing is used for training. The
two outer folds overlap, and the small search is not an exhaustive model
optimization. Scores are offline proxy measurements, not Kaggle scores.

## Commands and artifact locations

All commands below completed with return code 0:

```powershell
python -B -m tests.stage2_run --output artifacts/stage2/tests/unit_tests.json
python -B -m tvrec.stage2 prepare --config configs/stage2.json
python -B -m tvrec.stage2 run --config configs/stage2.json
python -B -m tvrec.stage2 check --config configs/stage2.json
python -B -m tests.stage2_independent
```

- Configuration: `configs/stage2.json`; dependencies: `requirements-stage2.txt`.
- Implementation: `tvrec/stage2/`; methodology: `STAGE2.md`.
- Final comparison:
  `artifacts/stage2/runs/comparison-20261022T130604523971Z/`.
- That directory contains `comparison.json`, `selection.json`, `als_grid.json`,
  configurations, provenance and a hash-verified manifest. Every model/fold
  directory contains `recommendations.parquet`, `per_user_metrics.parquet`,
  `resources.json` and `examples.json`, with IDs, titles and categories for
  the same two original users. ALS factors and sparse content artifacts are
  retained there.
- Tests: `artifacts/stage2/tests/unit_tests.json`.
- Run verification: `artifacts/stage2/verification.json`.
- Independent verification: `artifacts/stage2/checks/independent.json`.
- Machine-readable completion index: `artifacts/stage2/completion.json`.
