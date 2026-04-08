# ECG Arrhythmia Detection: GitHub-Friendly Summary

This is the short public-facing version of the project report.

It is written for GitHub, so it only keeps the parts that are easy for other people to read in the repo:
- what the project is doing
- what model we ended up with
- what techniques we used
- what the main results were
- what still needs work

It does not link to local-only experiment folders, because those `artifacts/` are not uploaded to GitHub.

## What This Project Is About

The goal is simple:

use ECG beats from the MIT-BIH arrhythmia dataset to detect ventricular beats with a model that is still realistic for edge or IoT deployment.

The important lesson from this project is that getting a high score was not the hard part.
The hard part was getting a score that still looks good after the testing setup becomes stricter and more realistic.

## Current Model, In Plain Language

The best-performing version so far is a two-layer idea:

1. `generic base model`
   A model that works on unseen patients.

2. `personalized calibration branch`
   A small extra branch that looks at earlier beats from the same record and asks:
   "What is normal for this person right now?"

### Current architecture

```text
ECG record
  |
  +--> R-peak aligned beat windows
          |
          +--> local beat neighborhood
          |      |
          |      +--> multiscale morphology encoder
          |      +--> short-range rhythm encoder
          |      +--> RR features
          |      +--> RR-normalized center beat
          |
          +--> causal history from the same record
                 |
                 +--> history morphology prototype
                 +--> rolling RR baseline
          |
          +--> fuse everything
          |
          +--> ventricular beat vs normal beat
```

## Main Techniques We Used

These are the main ideas that actually mattered:

- `multiscale morphology`
  Instead of using one fixed view of a heartbeat, the model looks at short, medium, and wider waveform patterns at the same time.

- `short-range rhythm context`
  A beat is easier to judge when the model can also see the beats right before and after it.

- `RR-aware features`
  The model does not only look at shape. It also looks at timing: whether the beat came early or late compared with the recent rhythm.

- `RR-normalized beat view`
  We also give the model a version of the beat that is rescaled to the local heart cycle, so it is less sensitive to different patient heart rates.

- `beat-balanced validation folds`
  This was one of the biggest fixes.
  Some records have many ventricular beats and some have almost none, so validation folds had to be balanced by beat counts, not just by record labels.

- `causal personalization`
  The personalized version only looks at earlier beats from the same record.
  It does not peek at future beats.

- `latency tracking`
  Every time the model got stronger, we also checked how much slower it became.

## Result Timeline

| Stage | Scope | Main result | What it told us |
|---|---|---|---|
| Random split quick check | Easy early setting | `baseline` best F1 `0.9922`, `attention` best F1 `0.9726` | High numbers were easy to get, but this setup was too optimistic. |
| First strict context model | de Chazal, fold 0, 3 seeds | mean `P 0.7739 / R 0.8693 / F1 0.8147` | Switching to unseen-patient testing made the problem much harder. |
| Beat-balanced context | de Chazal, 5 folds x 3 seeds | mean `P 0.7901 / R 0.9008 / F1 0.8348` | Better validation design made the result more realistic and more stable. |
| RR-aware generic model | de Chazal, 5 folds x 3 seeds | mean `P 0.7995 / R 0.9500 / F1 0.8629`, best single `F1 0.9420` | Explicit rhythm timing helped more than just adding more shape modeling. |
| Personalized model | de Chazal, 5 folds x 3 seeds | mean `P 0.8177 / R 0.9676 / F1 0.8817`, best single `F1 0.9517` | Personal history still helps, but the gain is smaller than it first looked in a single-seed scan. |

## Current Best Numbers

### Best generic setting

This is the main benchmark setting for unseen-patient generalization.

- model: `rr-context`
- scope: de Chazal inter-patient, `5 folds x 3 seeds`
- mean: `P 0.7995 / R 0.9500 / F1 0.8629`
- latency: `12.70 ms/window`
- best single saved run: `P 0.9166 / R 0.9689 / F1 0.9420`

### Best personalized setting

This is the "generic model + personal calibration" setting.

- model: `personalized-rr-context`
- scope: de Chazal inter-patient, `5 folds x 3 seeds`
- mean: `P 0.8177 / R 0.9676 / F1 0.8817`
- latency: `25.34 ms/window`
- best single saved run: `P 0.9265 / R 0.9783 / F1 0.9517`

## Personalized Ablation

We also split the personalized branch into its two main parts:

- `RR baseline`
  A short personal timing baseline.

- `history morphology prototype`
  A short personal "what your recent normal beat looks like" baseline.

### Fold 0, 3-seed comparison

| Variant | Mean F1 | Mean latency |
|---|---:|---:|
| Full personalized | `0.9266` | `27.20 ms` |
| RR-only | `0.9193` | `13.03 ms` |
| Prototype-only | `0.9214` | `26.74 ms` |

### What this means

- Using both together is best.
- The `history morphology prototype` seems to carry most of the accuracy gain.
- The `RR baseline` still helps, but the extra boost is smaller.
- Most of the added latency comes from the history morphology side, not from the RR baseline alone.

## What We Can Reasonably Claim Right Now

- Random split numbers were too optimistic and should not be treated as the main result.
- The stricter `de-chazal-interpatient` setting is the right main benchmark for a paper.
- `multiscale morphology + short-range rhythm context + RR-aware timing` is the best-performing generic direction so far.
- `personalized calibration` helps, but it should be reported as a separate setting because it uses personal history.
- Accuracy is no longer the only story; latency matters a lot for this project.

## What Still Needs Work

- The generic model is better than earlier versions, but some folds still show precision instability.
- The personalized branch helps on average, but it is not a free win because it roughly doubles latency.
- The personalized ablation has only been checked carefully on fold 0 so far.
- Reference baselines from earlier papers still need to be re-tested under the same strict split.

## Good Repo Entry Points

- [README.md](README.md)
- [RESULTS.md](RESULTS.md)
- [train.py](train.py)
- [sweep.py](sweep.py)
- [src/heart/data.py](src/heart/data.py)
- [src/heart/models.py](src/heart/models.py)
- [src/heart/train.py](src/heart/train.py)
