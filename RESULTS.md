# Prototype Results

## Current status

The prototype now has six model families:

- `baseline`
- `attention`
- `morph`
- `context`
- `rr-context`
- `personalized-rr-context`

and supports:

- random record-wise split
- de Chazal inter-patient split
- random DS1 validation
- record-stratified DS1 folds
- beat-balanced DS1 folds
- RR-feature extraction and RR-normalized beat views
- causal within-record history baselines for personalized experiments
- multi-seed aggregation through `sweep.py`

## Stricter summary

The most important new result is this:

- adding local rhythm context helped
- changing the DS1 validation fold construction helped even more
- adding RR-aware inputs helped again, especially on recall and cross-fold robustness
- adding causal history as an extra personalization signal still helps on the full 15-run mean, but the gain is smaller than the first seed-42-only scan suggested and the latency cost is large

The project now looks much more viable than it did before.

## Why the old result looked so strong

Random patient-wise splits can still look extremely good.

Examples:

- attention, seed 42:
  - Precision: `0.9872`
  - Recall: `0.9585`
  - F1: `0.9726`
  - Artifact: `artifacts/20260405_204533_attention/`
- baseline, seed 42:
  - Precision: `0.9948`
  - Recall: `0.9896`
  - F1: `0.9922`
  - Artifact: `artifacts/20260405_204613_baseline/`

These show the code can hit the target range, but they are not the best research setting.

## Strict context result before fixing DS1 folds

The first strict context sweep used record-stratified DS1 folds:

```powershell
python sweep.py --models context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

Saved summary:

- `artifacts/sweeps/20260405_220720_531060_summary.md`

This already showed promise:

- Precision mean: `0.7739`
- Recall mean: `0.8693`
- F1 mean: `0.8147`

Best single run:

- Precision: `0.9112`
- Recall: `0.9149`
- F1: `0.9131`

But once we checked more DS1 folds, some folds collapsed badly because the validation record mix was still misleading.

## What was wrong with the old DS1 folds

DS1 records have extremely uneven ventricular beat counts.

Examples:

- `208`: `992`
- `223`: `473`
- `119`: `444`
- `203`: `444`
- several records: `0` or `1`

So simply stratifying by "does this record contain any ventricular beats" is not enough.

One concrete failure:

- old fold 1 summary: `artifacts/sweeps/20260405_222826_095291_summary.md`
- old fold 1 mean F1: `0.4719`

That is not a model-only problem. It is mostly a validation-fold construction problem.

## New best evaluation setup

The new setup uses beat-balanced DS1 folds:

- same strict de Chazal DS1/DS2 split
- same context model
- same weighted sampler
- but DS1 validation folds are assigned to balance ventricular-beat counts across folds

Command:

```powershell
python sweep.py --models context --seeds 42 43 44 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

Saved combined summary:

- `artifacts/sweeps/20260405_231812_384030_context_beat_balanced_all_folds_summary.md`

Aggregate result across 15 runs:

- Precision mean: `0.7901`
- Recall mean: `0.9008`
- F1 mean: `0.8348`

This was the first result that felt strong enough to build on; the later RR-aware runs improved on top of it.

## By-fold view

Beat-balanced fold means:

- fold 0:
  - Precision mean: `0.7412`
  - Recall mean: `0.8984`
  - F1 mean: `0.8079`
- fold 1:
  - Precision mean: `0.6536`
  - Recall mean: `0.9145`
  - F1 mean: `0.7493`
- fold 2:
  - Precision mean: `0.8128`
  - Recall mean: `0.8942`
  - F1 mean: `0.8512`
- fold 3:
  - Precision mean: `0.8515`
  - Recall mean: `0.9021`
  - F1 mean: `0.8728`
- fold 4:
  - Precision mean: `0.8913`
  - Recall mean: `0.8946`
  - F1 mean: `0.8928`

The important comparison is fold 1:

- old fold 1 mean F1: `0.4719`
- beat-balanced fold 1 mean F1: `0.7493`

That is a major improvement in evaluation stability.

## RR-aware context results

The new `rr-context` model adds:

- explicit RR-interval features per beat
- a second beat view resampled to the local cardiac cycle
- the same local context branch as the earlier `context` model

The first controlled comparison was on beat-balanced fold 0 with 3 training seeds.

Command:

```powershell
python sweep.py --models context rr-context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

Saved summary:

- `artifacts/sweeps/20260406_041231_889091_summary.md`

Aggregate fold 0 result:

- `context`
  - Precision mean: `0.6871`
  - Recall mean: `0.9073`
  - F1 mean: `0.7678`
- `rr-context`
  - Precision mean: `0.8128`
  - Recall mean: `0.9386`
  - F1 mean: `0.8703`

So on the same strict fold, RR-aware modeling improved all three key metrics, with the biggest gain in precision.

## RR-aware cross-fold view

To check whether that gain was just a fold 0 effect, both models were also run across all 5 beat-balanced folds with seed 42.

Saved summaries:

- `context`: `artifacts/sweeps/20260406_043234_114488_summary.md`
- `rr-context`: `artifacts/sweeps/20260406_042407_775061_summary.md`

Cross-fold seed 42 aggregate:

- `context`
  - Precision mean: `0.7633`
  - Recall mean: `0.8814`
  - F1 mean: `0.8038`
  - Latency mean: `7.8474 ms/window`
- `rr-context`
  - Precision mean: `0.7845`
  - Recall mean: `0.9533`
  - F1 mean: `0.8496`
  - Latency mean: `10.9559 ms/window`

Fold-by-fold F1:

- fold 0: `0.8759 -> 0.9233`
- fold 1: `0.6295 -> 0.6669`
- fold 2: `0.8203 -> 0.8339`
- fold 3: `0.8429 -> 0.9231`
- fold 4: `0.8503 -> 0.9008`

So the RR-aware branch improved F1 on every fold in this seed 42 scan, and lifted recall above `0.94` on 4 of 5 folds.

To make the generic result more paper-ready, `rr-context` was then run across all 5 folds with 3 seeds:

```powershell
python sweep.py --models rr-context --seeds 42 43 44 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1 --no-export
```

Saved summary:

- `artifacts/sweeps/20260407_144513_434424_summary.md`

Cross-fold 15-run aggregate:

- `rr-context`
  - Precision mean: `0.7995`
  - Recall mean: `0.9500`
  - F1 mean: `0.8629`
  - Latency mean: `12.7047 ms/window`

Fold-by-fold F1 mean:

- fold 0: `0.9088`
- fold 1: `0.7823`
- fold 2: `0.8273`
- fold 3: `0.8720`
- fold 4: `0.9240`

This full multi-seed view is more realistic than the earlier seed 42 scan:

- `rr-context` is still stronger than `context` on the strict generic mean
- but the gain is not as clean as the seed 42 scan first suggested
- fold 1 to fold 3 still show meaningful seed-to-seed precision instability

## Best strict run so far

Best observed strict run so far:

- `rr-context`, beat-balanced fold 4, seed 43:
  - Precision: `0.9166`
  - Recall: `0.9689`
  - F1: `0.9420`
  - Artifact: `artifacts/20260407_144229_510756_rr-context/`

Compared with the poster target:

- Precision target: `0.88`
- Recall target: `0.94`
- F1 target: `0.91`

This run:

- beats precision
- beats recall
- beats F1

## Personalized history experiment

The idea here was:

- keep the strong `rr-context` branch
- add a second branch that only sees previous beats from the same record
- summarize those previous beats into a personal rhythm and morphology baseline
- avoid using patient ID directly

This is a personalized setting, not a pure generic benchmark setting.

### What the personalized branch uses

For each target beat, the new model sees:

- the same symmetric local context as `rr-context`
- RR features for that local context
- an RR-normalized view of the center beat
- a causal history window of previous beats from the same record
- rolling RR baseline statistics from that history
- a morphology prototype computed from the history embeddings

### What happened

The first version was too aggressive: it replaced the local context branch with a pure past-only branch, and it collapsed badly on the strict test split.

The improved version kept the original `rr-context` branch and only added causal history as an auxiliary personalization signal.

To check whether that was just a lucky single run, a 3-seed fold 0 sweep was added:

```powershell
python sweep.py --models personalized-rr-context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1 --history-beats 8
```

Saved summary:

- `artifacts/sweeps/20260406_114742_860732_summary.md`

Strict fold 0, seed 42, 3 epochs:

- `rr-context`
  - Precision: `0.7885`
  - Recall: `0.9761`
  - F1: `0.8723`
  - Latency: `10.84 ms/window`
  - Artifact: `artifacts/20260406_053847_992821_rr-context/`
- `personalized-rr-context`, `history-beats=8`
  - Precision: `0.9191`
  - Recall: `0.9171`
  - F1: `0.9181`
  - Latency: `22.72 ms/window`
  - Artifact: `artifacts/20260406_051652_439869_personalized-rr-context/`
- `personalized-rr-context`, `history-beats=4`
  - Precision: `0.8691`
  - Recall: `0.9090`
  - F1: `0.8886`
  - Latency: `20.78 ms/window`
  - Artifact: `artifacts/20260406_053351_858648_personalized-rr-context/`

3-seed fold 0 aggregate:

- `rr-context`
  - Precision mean: `0.8128`
  - Recall mean: `0.9386`
  - F1 mean: `0.8703`
  - Latency mean: `11.6759 ms/window`
  - Summary: `artifacts/sweeps/20260406_041231_889091_summary.md`
- `personalized-rr-context`, `history-beats=8`
  - Precision mean: `0.8983`
  - Recall mean: `0.9577`
  - F1 mean: `0.9266`
  - Latency mean: `27.2004 ms/window`
  - Summary: `artifacts/sweeps/20260406_114742_860732_summary.md`

A small cross-fold follow-up was then run to see whether the gain survives outside fold 0:

```powershell
python sweep.py --models personalized-rr-context --seeds 42 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1 --history-beats 8
```

Saved summary:

- `artifacts/sweeps/20260406_142915_054244_summary.md`

Cross-fold seed 42 aggregate:

- `rr-context`
  - Precision mean: `0.7845`
  - Recall mean: `0.9533`
  - F1 mean: `0.8496`
  - Latency mean: `10.9559 ms/window`
  - Summary: `artifacts/sweeps/20260406_042407_775061_summary.md`
- `personalized-rr-context`, `history-beats=8`
  - Precision mean: `0.8722`
  - Recall mean: `0.9652`
  - F1 mean: `0.9141`
  - Latency mean: `26.9671 ms/window`
  - Summary: `artifacts/sweeps/20260406_142915_054244_summary.md`

Fold-by-fold F1 in the seed 42 cross-fold scan:

- fold 0: `0.9233 -> 0.9272`
- fold 1: `0.6669 -> 0.9343`
- fold 2: `0.8339 -> 0.8377`
- fold 3: `0.9231 -> 0.9499`
- fold 4: `0.9008 -> 0.9212`

### Interpretation

This does suggest a real tradeoff, but the gain now looks more real than before:

- the personalized history branch can raise precision, recall, and F1 on this fold
- the gain is stable across 3 seeds, not just one run
- but it more than doubles CPU latency in the current implementation

The cross-fold seed 42 scan strengthened that conclusion:

- the gain is not restricted to fold 0
- in this first cross-fold check, `personalized-rr-context` improved F1 on all 5 folds
- fold 1 in particular improved dramatically, mostly by fixing the precision collapse seen in generic `rr-context`

To make that result more paper-ready, the same personalized setting was then run across all 5 folds with 3 seeds:

```powershell
python sweep.py --models personalized-rr-context --seeds 42 43 44 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1 --history-beats 8 --no-export
```

Saved summary:

- `artifacts/sweeps/20260407_195507_536570_summary.md`

Cross-fold 15-run aggregate:

- `rr-context`
  - Precision mean: `0.7995`
  - Recall mean: `0.9500`
  - F1 mean: `0.8629`
  - Latency mean: `12.7047 ms/window`
  - Summary: `artifacts/sweeps/20260407_144513_434424_summary.md`
- `personalized-rr-context`, `history-beats=8`
  - Precision mean: `0.8177`
  - Recall mean: `0.9676`
  - F1 mean: `0.8817`
  - Latency mean: `25.3438 ms/window`
  - Summary: `artifacts/sweeps/20260407_195507_536570_summary.md`

Fold-by-fold F1 mean in the full 15-run view:

- fold 0: `0.9088 -> 0.9399`
- fold 1: `0.7823 -> 0.8115`
- fold 2: `0.8273 -> 0.8665`
- fold 3: `0.8720 -> 0.8968`
- fold 4: `0.9240 -> 0.8937`

So this is still a stronger candidate than it first looked, but it is not an automatic replacement for `rr-context`.

It is more accurate to say:

- `rr-context` is still the lighter edge baseline
- personalized history is now a credible second setting for patient adaptation
- the full 15-run mean still improves over generic `rr-context`, but the advantage shrinks once multiple seeds are included
- fold 4 regresses and several fold/seed combinations still show precision collapse

## What we can now claim

The most defensible statement is:

- yes, the context model is a real improvement over the older models
- yes, adding RR-aware inputs improved the strict beat-balanced setup again
- yes, the project can reach poster-level F1 under a strict inter-patient setup
- yes, fold construction matters a lot for whether that conclusion looks stable
- yes, per-record history can help precision/recall/F1, but it changes the evaluation setting and increases latency
- yes, the full 5-fold x 3-seed picture is now available for both `rr-context` and `personalized-rr-context`
- yes, personalized history still improves the overall 15-run mean, but not on every fold and not by as much as the seed-42-only scan first suggested
- no, the personalized branch is not yet a drop-in replacement for the lighter generic edge model

## Things that helped

- stronger morphology backbone
- adding a small local rhythm-context branch
- adding RR features plus an RR-normalized beat view
- adding a causal personalized history baseline on top of `rr-context` when precision matters more than recall
- weighted sampler without extra class-weighted loss
- beat-balanced DS1 validation folds

## Things that did not help enough

- naive record-stratified DS1 folds
- longer training on unstable validation settings
- two-lead input with the current attention model
- larger beat window size `512`
- too wide a local context window (`context-radius 2`)
- a pure past-only personalized branch that removes the local context model

## Best next steps

If we keep going, the next highest-value steps are:

- replace "best single epoch" selection with a more stable validation policy across folds
- add clean ablations for the personalized branch: RR baseline only, morphology prototype only, and different `history-beats`
- run a stricter causal / cold-start evaluation so the personalization story matches real online deployment
- sweep smaller RR-aware rhythm encoders so the gain stays edge-friendly
- tighten reproducibility, because some same-setting reruns still move more than we would want for a paper
- retest representative reference baselines under the same strict split instead of comparing against random-split numbers
- only after that, revisit quantization or TurboQuant
