# SWEET.TV exam — calibrated recommendation experiments and final predictions

## Stage 3: designated first competition submission

The selected final method is **Personal History with Global Popularity fallback**.
It reuses `tvrec.stage2.rankers.PersonalHistory` unchanged, with a **90-day
half-life** and **0.25 duration coefficient**. Hyperparameters are frozen in
`configs/final_history_v1.json` and checked against the completed Stage 2 model
configuration. No final-label or Kaggle-score optimization is performed.

From the project root, with the original `data/` and verified Stage 1/2 artifacts
available at their recorded local paths:

```powershell
python -m pip install -r requirements.txt -r requirements-stage2.txt
python -B -m tvrec_final.testing --output artifacts/final/tests/unit_tests.json
python -B -m tvrec_final generate --config configs/final_history_v1.json
python -B -m tvrec_final verify --config configs/final_history_v1.json
```

`generate` writes **`submission_history_v1.csv` in the project root**. It contains
the exact header `user_id,tv_show_id`, all 1,260 template users as lossless strings
in original order, and five distinct positive space-separated programme IDs per
row. A different existing submission is never overwritten; identical regeneration
leaves the existing file intact. The command fits the selected model anew and
checks a second fit for byte-identical CSV output.

### Final training and availability contract

- All permitted historical evidence is aggregated as one window ending
  **exclusively at `2020-07-27 00:00:00 Europe/Kyiv`**, equivalent to
  `2020-07-26T21:00:00Z`. Weeks 31–42 viewing is not used.
- Personal training feedback remains Stage 2's **>=50% completed broadcasts**,
  with per-programme broadcast frequency, credited duration and recency. The
  **>80% rule is the primary evaluation proxy**, not a new training filter.
- Completion is computed per broadcast instance and continuous channel session;
  separate sittings/repeated broadcasts do not accumulate to qualify. Cutoff
  clipping changes the observed numerator, not the broadcast denominator.
  Channel switches and conflicting schedules retain the validated Stage 2 rules.
- Popularity counts distinct viewers with any positive historical support.
  Cold users receive this ranking; short histories are filled using the exact
  validated popularity-prior score policy. Previously viewed programmes remain
  eligible. Scores descend, with ascending programme ID for equal final scores.
- Final candidates are **only positive IDs appearing in
  `export_arh_31-42-final.csv`**. Future schedules provide known item availability
  and titles/categories for display; they provide no fitted viewing evidence.

### Final verification and artifacts

`verify` reads the root CSV from disk using an independent CSV/ranking verifier,
compares exact template strings/order, reads the actual future schedule,
reconstructs Top-5 from every candidate's saved score, checks finite scores, and
refits the unchanged model in a fresh process. The resulting CSV must be
byte-identical. It also verifies original input hashes/sizes/modification times,
prior manifests/artifacts, the frozen parameters, training bounds and Git ignores.

- `artifacts/final/manifest.json`: submission SHA-256, inputs, cutoff, method,
  candidate/cold-user counts, source/environment identities, measured execution
  time, tests, preservation and reproducibility evidence.
- `artifacts/final/verification.json`: independent disk readback and fresh-process
  refit result.
- `artifacts/final/runs/history-v1-<UTC timestamp>/`: frozen configurations,
  sparse History weights, all candidate scores, final recommendation Parquet,
  tests and checks. `examples_final.json` contains final Top-5 IDs/titles/categories
  for the same two preselected users; `example_histories.json` and
  `example_viewing_sessions.parquet` contain their cutoff-safe viewing histories.
  These are explicitly labelled **final competition predictions**, distinct from
  Stage 2's held-out validation examples.
- `artifacts/final/preservation_before.json` snapshots pre-existing files. The
  requested README/Git-ignore updates are documented exceptions with original
  copies retained under `artifacts/final/preservation/`. Existing experiment
  evidence and AI session records are not rewritten or deleted.

The final test runner writes only to `artifacts/final/`. The generation workflow
initializes a preservation snapshot if this is a new reproduction workspace.
Final-stage commands do not contact Kaggle or generate a PDF.

### Exam evaluation methodology and observed selection evidence

MAP@5 is macro-averaged over the fixed **1,260-user cohort**. AP@5 uses
`min(5, number_of_relevant_items)` as denominator, credits each ID once, and gives
empty relevance zero. All methods share users and candidate pools within a fold.
The primary label is **strictly >80% completion during one uninterrupted
same-channel interval**, following third-party reporting of the competition rule;
official hidden-label code is unverified. Continuous >=50% and original cumulative
>=60-second labels are retained as sensitivities.

Parameter/hybrid selection used an inner `[April 20, May 18)` Kyiv split. Frozen
comparisons used the main `[May 18, July 27)` and recent `[July 13, July 27)`
holdouts. History's primary MAP@5 was **0.509021 / 0.361846**, compared with
Popularity's **0.439197 / 0.250356**. The History/ALS hybrid scored
**0.513351 / 0.356853**; its benefit was inconsistent, supporting History as the
designated first submission. ALS and Content-Based remain separately evaluated
exam methods. These are offline proxy scores, not Kaggle public/private scores.
The outer periods overlap and are not statistically independent. Device IDs are
absent, so concurrent-channel logs require the documented conservative policy.

See [STAGE2.md](STAGE2.md) for algorithms and
[STAGE2_RESULTS.md](STAGE2_RESULTS.md) for the full observed comparison.

### Version control

`.gitignore` excludes `data/`, credentials (`.kaggle/`, `kaggle*.json`, `.env*`,
tokens/keys), caches, and large artifacts/model archives. The small root
**`submission_history_v1.csv` is explicitly not ignored** and is ready to add to
Git. The workflow tests ignore rules with a disposable Git directory without
creating, committing to, or changing the workspace repository. Keep original AI
session records in their existing location; do not include secrets in commits.

## Stage 1 temporal recommendation foundation

Reproduction target: **Python 3.10.11**. Runtime libraries are pinned in
`requirements.txt` and `pyproject.toml`. The standard-library `unittest` runner
is used; pytest, implicit ALS and content models are not required for Stage 1.

```powershell
python -m pip install -r requirements.txt
python -m tests.run --output artifacts/tests/unit_tests.json
python -m tvrec run --config configs/default.json --fold main
python -m tvrec check-run --config configs/default.json
```

The package runs directly from the repository root. An optional editable install
(`python -m pip install -e . --no-build-isolation --no-deps`) also provides `tvrec`.
The Stage 1 commands do not write to `data/`, export a submission, or contact Kaggle.
Their original output locations below describe Stage 1 reproduction; completed
Stage 1/2 evidence is preserved by running the separate final-stage commands above.

## Policies

- External user IDs are exact strings. Template order is preserved end to end.
- Naive timestamps are interpreted with `zoneinfo.ZoneInfo("Europe/Kyiv")`.
  This is an evidence-based assumption, not a documented source timezone.
  Ambiguous/nonexistent local timestamps fail explicitly. Arithmetic uses UTC.
- Viewing rows are deduplicated, then overlapping or touching intervals are
  unioned within each user/channel. Different channels are not merged.
- Broadcast timelines are disjoint canonical segments. Simultaneous **distinct
  labels**, including unknown label 0, receive equal shares. Positive labels
  cannot gain more credit than actual viewing time. Zero/missing programme IDs
  are unknown and never become items. Nonpositive intervals are excluded and
  counted. Channel IDs are matched exactly; no mappings are invented.
- The indexed join uses two binary searches into each channel's disjoint
  timeline, expands only actual overlaps, and writes bounded Parquet batches.
  Evidence retains exact UTC endpoints, allocation denominators, credited
  seconds, normalized view IDs, and segment IDs.
- Temporal clipping precedes aggregation. `seconds` is uncapped evidence;
  `confidence_seconds` optionally caps its user/programme aggregate. Relevance
  defaults to >=60 cumulative credited seconds and is configurable.
- Main training ends exclusively at May 18, 2020, 00:00 Kyiv. Validation ends
  exclusively at July 27. Viewing beginning in the test period is excluded;
  any session crossing its beginning is clipped.
- Main validation candidates are all positive IDs from the 21–30 schedule
  file; final catalogue IDs come from 31–42. Candidate selection never reads
  held-out consumption. Previously viewed programmes are allowed.
- Default popularity counts distinct users with **any positive training
  credited seconds**, over all training history. This support rule is separate
  from the relevance threshold. Optional lookback days and minimum support
  seconds are explicit. Ties use ascending programme ID. The global ranking is
  also the cold-user fallback; unseen candidates have zero support.
- AP@5 credits each relevant ID once, with denominator min(5, |relevance|).
  Empty relevance contributes zero. Both the fixed template cohort and the
  evaluable cohort are reported. Official hidden-label construction and
  competition-specific duplicate/order constraints remain unverified.

## Artifacts and caching

`artifacts/latest_run.json` identifies the completed run directory, containing
metrics, diagnostics, validation-only recommendation Parquet, per-user AP,
preselected examples, candidate IDs, input fingerprints, checks and a resource
manifest. `check-run` verifies output/input hashes and independently recomputes
metrics from saved recommendations and cached validation interactions.

- `artifacts/cache/<key>/`: timestamped positive evidence, normalized viewing,
  source-timestamped broadcast metadata, catalogue-reference programme metadata,
  candidates and diagnostics. Cache identities include input content hashes,
  preprocessing configuration, timezone-data hash, relevant source-code hashes,
  schemas and library versions. A completion manifest and artifact hashes are
  verified before reuse. Failed builds cannot be mistaken for complete caches.
- `artifacts/windows/<key>/`: window-clipped user/programme seconds, fragment
  counts, distinct normalized-view counts, confidence seconds and relevance.
  Keys add window endpoints, threshold, cap and aggregation code identity.

All-schedule `programme_metadata.parquet` is a **catalogue reference**, not a
historically fitted feature table. `historical_metadata(..., cutoff)` filters
timestamped broadcasts before the cutoff. Popularity never uses title, genre,
future airtime or other metadata signals. Example titles use training-as-of
metadata. A later content implementation must use the temporal metadata API.

Raw-row audit coverage is retained separately from normalized interval
coverage: deduplication, viewing union, and ambiguous-label allocation alter
the model-evidence denominator and duration totals. The audit comparison
records measured differences rather than replacing those denominators.

## Other folds

```powershell
python -m tvrec list-folds --config configs/default.json
python -m tvrec evaluate --config configs/default.json --fold recent_a
python -m tvrec evaluate --config configs/default.json --fold recent_b
```

These reuse the full join and aggregate cached UTC evidence for their windows.
They are interfaces for later experiments; Stage 1's real-data run evaluates
only `main`. A twelve-week stress fold is configured as well. The folds overlap
and should not be represented as statistically independent holdouts.

Preselected example IDs are fixed in configuration before scoring. Stage 1
popularity intentionally returns the same ranking to every user. The pure CSV
formatter is unit-tested in memory; the CLI does not export a Kaggle CSV.
