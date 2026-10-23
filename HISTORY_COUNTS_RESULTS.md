# Focused broadcast-count experiment: verified negative result

## Decision

Retain **Personal History v1**. Every tested finalist scored lower on the recent
outer fold. The frozen inner winner improved the main fold but lost on the recent
fold, so **no `submission_history_v2.csv` was generated**.

All numbers below are offline fixed-cohort MAP@5 using the existing strict
greater-than-80% continuous-completion relevance labels. Kaggle scores were not
used for selection. All original project files and artifacts remain unchanged.

## Protocol and exact rules

- Reused the verified Stage 2 continuous-session/broadcast-instance cache.
- Used the exact existing 1,260 template users, candidate ID lists and labels.
- Inner selection: April 20 to May 18, 2020, Europe/Kyiv. Outer evaluation:
  May 18 to July 27 (main), and July 13 to July 27 (recent). All windows are
  half-open; the outer periods overlap and are not independent.
- Eight inexpensive inner trials: strict 30/50/80% counts; two count mixtures
  `(0.25, 0.50, 0.25)` and `(0.10, 0.30, 0.60)`; strict-50 counts with the
  existing duration/recency factors; and two upcoming-airing exponents
  `0.25` and `0.50`. Six family finalists were frozen before outer evaluation.
- Predeclared acceptance required the inner winner to gain at least `0.005`
  MAP@5 on both outer folds and have both descriptive paired-bootstrap lower
  bounds above zero. The result fails even a simple positive-gain-on-both rule.

Let `C_t(u,i)` count distinct past broadcast instances of programme `i` for which
user `u` watched **strictly more than** fraction `t` in any one continuous sitting.
Separate sittings are never summed to qualify one broadcast; repeated qualifying
sittings still count that broadcast only once. Ambiguous schedule seconds are
excluded, viewing numerators are clipped to the training window, and denominators
remain full UTC broadcast durations. Channel switches use the existing
truncate-without-resuming policy.

Let `S50` be the sum, over strict-50 qualifying broadcasts, of the maximum
single-sitting observed seconds, and `age50` the number of days since the most
recent qualifying sitting. Define
`D50 = (1 + 0.25 * ln(1 + S50/3600)) * 2**(-age50/90)`.
Define the inclusive equivalents `C_ge50`, `S_ge50` and `D_ge50` for v1.

| Method | Personal ranking value before the popularity prior |
|---|---|
| History v1, unchanged | `ln(1 + C_ge50) * D_ge50` |
| Count >30% | `C_0.30` |
| Count >50% | `C_0.50` |
| Count >80% | `C_0.80` |
| Selected mixture | `0.25*C_0.30 + 0.50*C_0.50 + 0.25*C_0.80` |
| Count >50% + signals | `C_0.50 * D50` |
| Count >50% + airings | `C_0.50 * (1 + F_i)**0.25` |

`F_i` is the number of distinct scheduled broadcasts overlapping the upcoming
validation window, deduplicated by channel, programme, start and end. It uses
schedule metadata only. Thus the direct reference baseline differs from v1 in
three ways: raw counts instead of log-counts, strict >50% instead of inclusive
>=50%, and removal of duration/recency weighting.

All variants use the original distinct-historical-viewer popularity definition
for fallback. Warm-user scores add `1e-7 * P_i / max(1, max(P))`; fully cold users
receive popularity scores. Unseen candidate slots are filled by popularity.
Equal final scores use ascending programme ID. Count variants use float64 so the
small popularity tie-break remains effective for integer counts; v1 is reused
with its original float32 scoring unchanged. Previously viewed programmes remain
eligible.

## Outer comparison

Runtimes are measured scoring/fit, ranking, coverage and evaluation time on cached
features, excluding common feature preparation and artifact writes.

| Variant | Main MAP@5 | Delta v1 | Recent MAP@5 | Delta v1 | Seconds, main/recent |
|---|---:|---:|---:|---:|---:|
| History v1 | 0.509021 | 0.000000 | 0.361846 | 0.000000 | 0.350 / 0.222 |
| Count >30% | 0.519956 | +0.010935 | 0.350964 | -0.010882 | 0.339 / 0.193 |
| Count >50% | 0.518002 | +0.008981 | 0.357892 | -0.003955 | 0.328 / 0.191 |
| Count >80% | 0.501828 | -0.007193 | 0.353880 | -0.007967 | 0.328 / 0.180 |
| Mixture 25/50/25 | 0.518279 | +0.009259 | 0.355024 | -0.006823 | 0.360 / 0.189 |
| Count >50% + signals | 0.508913 | -0.000108 | 0.359470 | -0.002376 | 0.320 / 0.191 |
| Count >50% + airings^0.25 | 0.522062 | +0.013041 | 0.353509 | -0.008337 | 0.340 / 0.205 |

The inner winner was **Count >50% + airings^0.25**, with inner MAP@5
`0.469103835978836` versus v1's `0.4469600970017637`. The selected mixture's
inner score was `0.46530158730158727`; the alternative mixture scored
`0.45984435626102294`. Airing exponent `0.50` scored `0.46081613756613754`.
Full-precision results for all inner trials are retained in `selection.json`.

For the frozen winner, paired user-bootstrap 95% delta intervals (2,000
replicates, seed 42) were `[+0.006367, +0.019855]` on main and
`[-0.014759, -0.001237]` on recent. These are descriptive within-fold intervals,
not independent cross-fold confirmation.

## Candidate coverage and cold starts

Every method produced five distinct positive eligible IDs for all **1,260 users**.
Candidate pools were exactly **6,363 main / 2,697 recent**. Table entries are
main/recent; "supported IDs" means distinct candidates with personal support
among the template users, and "cold" means no qualifying personal history.

| Method | Supported IDs | Cold users | Users without personal candidate support | Users with <5 supported candidates | Fallback positions |
|---|---:|---:|---:|---:|---:|
| History v1 | 1813 / 1368 | 15 / 6 | 15 / 7 | 40 / 21 | 121 / 69 |
| Count >30% | 1950 / 1427 | 12 / 4 | 12 / 5 | 27 / 15 | 94 / 54 |
| Count >50% | 1813 / 1367 | 15 / 6 | 15 / 7 | 40 / 21 | 121 / 69 |
| Count >80% | 1515 / 1231 | 20 / 10 | 22 / 12 | 83 / 59 | 249 / 168 |
| Mixture 25/50/25 | 1950 / 1427 | 12 / 4 | 12 / 5 | 27 / 15 | 94 / 54 |
| Count >50% + signals | 1813 / 1367 | 15 / 6 | 15 / 7 | 40 / 21 | 121 / 69 |
| Count >50% + airings | 1813 / 1367 | 15 / 6 | 15 / 7 | 40 / 21 | 121 / 69 |

## Tests, independent verification and reproducibility

- **106/106 tests passed**, including all previous 95, with no failures, errors
  or skips, in `5.1387 s`. New checks cover strict threshold boundaries,
  distinct-instance counting, disjoint sittings, duplicates/chunks, cutoff
  clipping, ambiguity exclusion, cold fallback, schedule windows and randomized
  maximum-single-sitting reference cases.
- Reconstructed inclusive-50 and strict-80 features exactly matched the prior
  verified tables on inner/main/recent/full windows for the template users.
- Unchanged v1 reproduced archived outer recommendation IDs and scores exactly.
- Independent disk verification recomputed **28,980 per-user AP values** using
  rational arithmetic across **23 model/fold outputs**, and checked exact users,
  order, candidate membership, distinct positive IDs and finite scores.
- A fresh-process replay rebuilt every one of those 23 rankings and per-user
  metric tables with **byte-identical Parquet outputs** and exactly equal
  aggregate metrics and coverage; replay took `7.1356 s`.
- All **476 pre-existing files** retain their SHA-256, size and modification
  time; the prior artifact inventory and original AI-session backup also match.
- Measured experiment time: **15.2563 s**, including `3.8150 s` for the one cached
  event scan; Windows process peak working set: **668,983,296 bytes**.

Commands executed successfully from `C:\Recsys\rufan_exam`:

```powershell
python -B -m tvrec_counts test --config configs/history_counts_v2.json
python -B -m tvrec_counts run --config configs/history_counts_v2.json
python -B -m tvrec_counts verify --config configs/history_counts_v2.json
python -B artifacts/history_v2/replay.py
```

Implementation: `tvrec_counts/`; configuration: `configs/history_counts_v2.json`.
Measured run:
`artifacts/history_v2/runs/counts-20261023T204311493666Z/`.
Verification: `artifacts/history_v2/verification.json`.
Replay evidence: `artifacts/history_v2/reproducibility.json`.
Completion index and evidence hashes: `artifacts/history_v2/completion.json`.

Recommended CSV: **`C:\Recsys\rufan_exam\submission_history_v1.csv`**.
Its unchanged SHA-256 is
`ae6ead5610f889d3edb067fdd54e3264dfdca5231993c8f8104f712503c11838`.
No v2 CSV was generated, no Kaggle upload was performed, and no PDF was created.

## External methodological disclosure

Nikita Melentev (2021),
[Two-layered Recommender System Methodology: A Prize-Winning Solution](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156).
[Accessible article](https://towardsdatascience.com/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156/).
Used as methodological inspiration for continuous-completion broadcast counts
and availability filtering. Implementation was developed locally from the
existing project evidence; no external solution code, trained models or
predictions were obtained or copied. The article is a third-party account,
not verified official hidden label-generation code.
