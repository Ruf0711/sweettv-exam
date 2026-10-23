# Stage 2: calibration and fair model comparison

Stage 1 files and artifacts are immutable. All Stage 2 artifacts live under
`artifacts/stage2/`; its separate CLI does not update Stage 1's latest-run pointer.

```powershell
python -m pip install -r requirements-stage2.txt
python -B -m tests.stage2_run --output artifacts/stage2/tests/unit_tests.json
python -B -m tvrec.stage2 prepare --config configs/stage2.json
python -B -m tvrec.stage2 run --config configs/stage2.json
python -B -m tvrec.stage2 check --config configs/stage2.json
python -B -m tests.stage2_independent
```

Observed results and the recommended strategy are in [STAGE2_RESULTS.md](STAGE2_RESULTS.md).

## Relevance calibration

The 2021-03-20 article by Nikita Melentev reports that the original challenge
considered a programme watched after **over 80% without changing channels**.
This is third-party reporting, not verified official label-generation code.
Only the rule description was consulted; no solution code, trained model or
prediction file was obtained. Source provenance is recorded in each run.

The primary proxy is strict `>80%`. Inclusive `>=50%` and the exact Stage 1
cumulative `>=60 seconds` proxy are retained as sensitivities. Completion uses
one continuous session and one distinct scheduled broadcast instance. It never
adds separate sittings or repeated broadcasts to reach a completion threshold.
Multiple qualifying sittings count a user/broadcast only once.

UTC elapsed broadcast duration is the denominator, including at cutoffs.
The observed numerator is clipped to the allowed window before qualification.
All broadcast conflicts, including conflicts with unknown ID 0, have their
ambiguous seconds excluded. Unknown IDs never become items. Source duplicates
are removed before broadcast-instance identification.

Raw logs have no device identifiers. A later different-channel start truncates
the previous channel run, and an old overlapping session does not automatically
resume. Exact same-time, different-channel starts create an unknown break.
Overlapping/touching same-channel records merge only within a channel run.
This is an explicit conservative reconstruction; concurrent devices cannot be
resolved exactly from these logs. Stage 1's independent cumulative semantics
remain unchanged.

## Model inputs and temporal selection

- Global Popularity retains Stage 1's distinct-viewer, any-positive-second rule.
- Personal History uses pre-cutoff completed-broadcast (`>=50%`) frequency,
  duration and recency. Weights are `log1p(frequency) *
  (1 + 0.25*log1p(seconds/3600)) * 2**(-age_days/90)`.
- Sparse ALS uses the compiled `implicit` CPU CSR implementation, fitting the
  implicit weighted least-squares objective, with nonzero confidence
  `1 + scale*log1p(history_weight)`, missing-item confidence 1, seed 42 and one
  thread. Eight factor/regularization/confidence combinations are tested.
- Content-Based uses training-only character TF-IDF titles and sparse metadata
  features, then sparse recency-weighted user profiles. Vocabulary, IDF and
  metadata dictionaries fit only pre-cutoff documents. Upcoming metadata for
  cold items is transformed with these frozen transformations. Profiles keep
  at most 1024 features per user. No validation viewing enters a profile.

All personalized methods use the same pre-cutoff completion feedback; Global
Popularity intentionally stays the original benchmark. No previously viewed
programme is filtered. Every model uses identical candidate IDs and the fixed
template-user order. Cold users receive global popularity. ALS cold items have
no fabricated factors; popularity supplies fallback/tie scores. Content-based
features can score unseen items sharing known metadata.

The small ALS grid and optional reciprocal-rank-fusion hybrid are selected on
an **inner temporal split**: train before April 20, validate April 20–May 17.
Parameters are frozen before the main May 18–July 26 evaluation and the recent
July 13–26 evaluation. The recent fold provides a shorter, less accumulated
relevance horizon. It overlaps the main evaluation period, so the outer fold
scores are not statistically independent. Hybrid trials occur only after
component evaluation; a hybrid is carried forward only if its inner score
exceeds the best individual by the predeclared margin.

Each run contains per-user metrics for all three proxies, recommendation
Parquet files, examples for the same two preselected users, model parameters,
ALS factors/loss traces, content transformations, resource measurements and
hash-verified manifests. No final-test recommendation or Kaggle CSV is created.

The Stage 2 test wrapper invokes the unchanged original runner and supplements
its report with a recursive Stage 2 source hash and dependency versions. Run
verification checks that the application code, all referenced evidence/window
caches and every model artifact still match their recorded hashes. Per-model
RSS is shared-process resident memory, not isolated incremental model memory;
the Windows OS peak is explicitly the lifetime process peak. Hybrid timing
includes the measured component fits/scores plus the separately measured fusion.
The independent checker rebuilds every completion table from UTC endpoint
arithmetic, checks broadcast/session bounds and source schedule membership,
reconstructs the two example users' sessions directly from original logs, and
recalculates every per-user AP using exact rational arithmetic.
