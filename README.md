# ECG Arrhythmia Prototype

An edge-oriented MIT-BIH ventricular arrhythmia prototype built to answer one question quickly:
can we match or beat the poster baseline before investing in a paper-ready system?

There are two ways to use this repository:

- **Part 1: Run released weights on Raspberry Pi 3/4/5.** Use this if you want to download the pretrained model and processed data, then immediately evaluate F1 and CPU inference speed.
- **Part 2: Train from source.** Use this if you want to rebuild the dataset, train models, run sweeps, and reproduce experiments.

Project summaries:

- [GitHub-friendly summary](GITHUB_REPORT.md)
- [Full experiment report](full_experiment_report.html)
- [GitHub Pages report](https://hengchunsong.github.io/ECG-arrhythmia-detection/)

The full experiment report preserves the detailed result timeline. Some old artifact links inside that report point to local experiment folders, while the released runnable weights and processed data are hosted under `v0.1.0-weights`.

Current implementation includes:

- automatic MIT-BIH download for source training
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

## Part 1: Run Released Weights On Raspberry Pi

Use this path when you do not want to retrain. It downloads:

- pretrained `rr-context` weights
- the matching threshold/metrics file
- a processed MIT-BIH cache with beat windows, RR features, and RR-normalized views

The release is here:
[v0.1.0-weights](https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/tag/v0.1.0-weights).

### 1. Clone And Install

On Raspberry Pi 3/4/5:

```bash
git clone https://github.com/HengchunSong/ECG-arrhythmia-detection.git
cd ECG-arrhythmia-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model And Processed Data

Run this from the repository root:

```bash
mkdir -p data/processed
curl -L -o rr-context-best.pt https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/rr-context-best.pt
curl -L -o rr-context-metrics.json https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/rr-context-metrics.json
curl -L -o data/processed/mitdb_binary_v2_w256_leads0_n48.npz https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/mitdb_binary_v2_w256_leads0_n48.npz
```

### 3. Run Inference And Measure Speed

For Raspberry Pi 4/5:

```bash
python scripts/evaluate_pretrained.py --model rr-context --weights rr-context-best.pt --metrics-json rr-context-metrics.json --data-root data --batch-size 128 --benchmark-runs 300
```

For Raspberry Pi 3, use a smaller batch:

```bash
python scripts/evaluate_pretrained.py --model rr-context --weights rr-context-best.pt --metrics-json rr-context-metrics.json --data-root data --batch-size 32 --benchmark-runs 100
```

The output includes both accuracy metrics and speed:

```json
{
  "precision": 0.9165,
  "recall": 0.9689,
  "f1": 0.9420,
  "latency_ms_per_window_cpu": 11.21,
  "windows_per_second_cpu": 89.18
}
```

`latency_ms_per_window_cpu` means how many milliseconds one ECG window takes for one CPU inference.

### Optional Personalized Model

Download the personalized checkpoint:

```bash
curl -L -o personalized-rr-context-best.pt https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/personalized-rr-context-best.pt
curl -L -o personalized-rr-context-metrics.json https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/personalized-rr-context-metrics.json
```

Run it:

```bash
python scripts/evaluate_pretrained.py --model personalized-rr-context --weights personalized-rr-context-best.pt --metrics-json personalized-rr-context-metrics.json --data-root data --batch-size 64 --benchmark-runs 300
```

This model uses a causal history branch. It usually improves the best single-run result, but it is slower than `rr-context`.

### Windows PowerShell Download

If you are testing on Windows:

```powershell
New-Item -ItemType Directory -Force data\processed
Invoke-WebRequest -Uri https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/rr-context-best.pt -OutFile rr-context-best.pt
Invoke-WebRequest -Uri https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/rr-context-metrics.json -OutFile rr-context-metrics.json
Invoke-WebRequest -Uri https://github.com/HengchunSong/ECG-arrhythmia-detection/releases/download/v0.1.0-weights/mitdb_binary_v2_w256_leads0_n48.npz -OutFile data\processed\mitdb_binary_v2_w256_leads0_n48.npz
```

## Part 2: Train From Source

Use this path when you want to rebuild the dataset and train models yourself. If the processed cache is not present, the training script downloads MIT-BIH through `wfdb` and builds the cache.

### Install

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
