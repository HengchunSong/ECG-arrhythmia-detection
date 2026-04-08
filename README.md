# ECG Arrhythmia Prototype

An edge-oriented MIT-BIH ventricular arrhythmia prototype built to answer one question quickly:
can we match or beat the poster baseline before investing in a paper-ready system?

For a GitHub-friendly project summary, see [GITHUB_REPORT.md](GITHUB_REPORT.md).

Current implementation includes:

- automatic MIT-BIH download
- patient-wise record splits
- R-peak aligned beat windows
- binary labels: `normal=0`, `ventricular=1`
- a lightweight `baseline` 1D CNN
- an `attention` model with SE blocks and temporal self-attention
- a stronger `morph` model with multiscale morphology blocks
- a `context` model that combines beat morphology with nearby rhythm context
- an `rr-context` model that adds RR-interval features and an RR-normalized beat view
- a `personalized-rr-context` model that adds a causal per-record history baseline on top of `rr-context`
- TorchScript export and a lightweight dynamic quantization checkpoint
- multi-seed sweep support for more stable comparisons

## Install

```powershell
pip install -r requirements.txt
```

## Quick smoke test

This is only for checking that download, preprocessing, training, and evaluation all run end to end.

```powershell
python train.py --model attention --epochs 1 --records-limit 8 --max-beats-per-record 96
```

## Full training

Recommended first comparison:

```powershell
python train.py --model attention --epochs 20 --batch-size 512 --sampler none --class-weight balanced
```

Baseline comparison:

```powershell
python train.py --model baseline --epochs 20 --batch-size 512 --sampler none --class-weight balanced
```

Stronger morphology backbone:

```powershell
python train.py --model morph --epochs 20 --batch-size 512 --sampler weighted --class-weight none
```

Morphology plus nearby rhythm context:

```powershell
python train.py --model context --context-radius 1 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

RR-aware rhythm context:

```powershell
python train.py --model rr-context --context-radius 1 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

RR-aware rhythm context plus a causal history baseline:

```powershell
python train.py --model personalized-rr-context --context-radius 1 --history-beats 8 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

## Standard split option

For a more paper-friendly setup, use the classic de Chazal inter-patient split:

```powershell
python train.py --model attention --split-policy de-chazal-interpatient --split-seed 42 --epochs 20 --batch-size 512 --sampler none --class-weight balanced
```

Fold-based DS1 validation:

```powershell
python train.py --model morph --split-policy de-chazal-interpatient --de-chazal-val-mode fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 20 --batch-size 512 --sampler weighted --class-weight none
```

Context model on the stricter split:

```powershell
python train.py --model context --context-radius 1 --split-policy de-chazal-interpatient --de-chazal-val-mode fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

RR-aware context model on beat-balanced folds:

```powershell
python train.py --model rr-context --context-radius 1 --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

Personalized RR-aware model on beat-balanced folds:

```powershell
python train.py --model personalized-rr-context --context-radius 1 --history-beats 8 --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

Beat-balanced DS1 folds:

```powershell
python train.py --model context --context-radius 1 --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 20 --batch-size 256 --sampler weighted --class-weight none
```

Dual-lead variant:

```powershell
python train.py --model attention --leads 0 1 --split-policy de-chazal-interpatient --split-seed 42 --epochs 20 --batch-size 512 --sampler none --class-weight balanced
```

Notes:

- `--split-seed` controls the validation carve-out while keeping train/test policy fixed
- `--seed` controls training randomness
- `de-chazal-interpatient` uses the official DS1/DS2 record lists for test-time comparison
- `--de-chazal-val-mode fold` replaces the fragile random DS1 validation subset with a fixed stratified fold
- `--de-chazal-val-mode beat-balanced-fold` greedily balances ventricular-beat counts across DS1 validation folds
- `--context-radius 1` means the model sees the center beat plus one neighbor on each side
- `rr-context` adds per-beat RR features plus a resampled beat view normalized to the local cardiac cycle
- `personalized-rr-context` keeps that branch and adds a causal baseline from the previous `--history-beats` beats of the same record

## Multi-seed sweep

Run several seeds and save an aggregate summary:

```powershell
python sweep.py --models attention baseline --seeds 42 43 44 -- --split-policy de-chazal-interpatient --split-seed 42 --epochs 3 --batch-size 512 --sampler none --class-weight balanced
```

Current best robustness-oriented comparison:

```powershell
python sweep.py --models attention morph --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 512 --sampler weighted --class-weight none
```

Current best context comparison:

```powershell
python sweep.py --models context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

RR-aware fold comparison:

```powershell
python sweep.py --models context rr-context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

Personalized fold comparison:

```powershell
python sweep.py --models rr-context personalized-rr-context --seeds 42 43 44 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --de-chazal-val-fold 0 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1 --history-beats 8
```

Cross-fold beat-balanced context comparison:

```powershell
python sweep.py --models context --seeds 42 43 44 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

Cross-fold RR-aware scan:

```powershell
python sweep.py --models rr-context --seeds 42 --folds 0 1 2 3 4 -- --split-policy de-chazal-interpatient --de-chazal-val-mode beat-balanced-fold --de-chazal-num-folds 5 --split-seed 42 --epochs 3 --batch-size 256 --sampler weighted --class-weight none --context-radius 1
```

The sweep writes:

- `artifacts/sweeps/*_summary.json`
- `artifacts/sweeps/*_summary.md`

## Outputs

Each training run writes to `artifacts/<timestamp>_<model>/`:

- `best.pt`: best checkpoint
- `best.ts`: TorchScript export
- `best_dynamic_q.pt`: dynamic quantization checkpoint
- `metrics.json`: validation and test metrics
- `config.json`: run configuration

## Notes

- default dataset cache lives under `data/raw/mitdb/` and `data/processed/`
- default window size is `256` samples on lead list `0`
- use `--leads 0 1` to enable both MIT-BIH channels
- only normal and ventricular beats are kept
- processed cache also stores RR features and an RR-normalized beat view
- `--max-beats-per-record` is for smoke tests only; do not use it for meaningful scoring
- local CPU latency is useful for relative comparison, not for Raspberry Pi claims
