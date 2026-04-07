from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .train import parse_args, run_experiment


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed ECG experiments and aggregate metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "attention", "morph", "context", "rr-context", "personalized-rr-context"],
        default=["attention"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--export-runs", action="store_true")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("artifacts") / "sweeps",
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def strip_remainder_prefix(items: list[str]) -> list[str]:
    if items and items[0] == "--":
        return items[1:]
    return items


def build_run_args(
    train_args: list[str],
    model: str,
    seed: int,
    export_runs: bool,
    fold: int | None = None,
) -> argparse.Namespace:
    args_list = list(train_args)
    args_list.extend(["--model", model, "--seed", str(seed)])
    if fold is not None:
        args_list.extend(["--de-chazal-val-fold", str(fold)])
    if not export_runs and "--no-export" not in args_list:
        args_list.append("--no-export")
    return parse_args(args_list)


def aggregate_rows_by_model(rows: list[dict[str, object]]) -> dict[str, object]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    aggregate: dict[str, object] = {}
    for model, model_rows in by_model.items():
        metrics = {}
        for key in ["accuracy", "precision", "recall", "f1", "latency_ms_per_window_cpu"]:
            values = np.asarray([float(row[key]) for row in model_rows], dtype=np.float64)
            metrics[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        aggregate[model] = {
            "num_runs": len(model_rows),
            "metrics": metrics,
        }
    return aggregate


def summarize_runs(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"runs": rows}
    summary["by_model"] = aggregate_rows_by_model(rows)

    fold_values = sorted({int(row["de_chazal_val_fold"]) for row in rows})
    by_fold: dict[str, object] = {}
    for fold in fold_values:
        fold_rows = [row for row in rows if int(row["de_chazal_val_fold"]) == fold]
        by_fold[str(fold)] = {
            "num_runs": len(fold_rows),
            "by_model": aggregate_rows_by_model(fold_rows),
        }
    summary["by_fold"] = by_fold
    return summary


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Sweep Summary",
        "",
        "## Aggregates",
        "",
    ]
    by_model = summary["by_model"]
    for model, payload in by_model.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"- runs: `{payload['num_runs']}`")
        for metric_name, metric_stats in payload["metrics"].items():
            lines.append(
                f"- {metric_name}: "
                f"mean=`{metric_stats['mean']:.4f}` std=`{metric_stats['std']:.4f}` "
                f"min=`{metric_stats['min']:.4f}` max=`{metric_stats['max']:.4f}`"
            )
        lines.append("")

    by_fold = summary.get("by_fold", {})
    if by_fold:
        lines.append("## By Fold")
        lines.append("")
        for fold, fold_payload in by_fold.items():
            lines.append(f"### fold {fold}")
            lines.append("")
            for model, payload in fold_payload["by_model"].items():
                lines.append(f"- {model} runs: `{payload['num_runs']}`")
                lines.append(
                    f"- {model} f1: mean=`{payload['metrics']['f1']['mean']:.4f}` "
                    f"std=`{payload['metrics']['f1']['std']:.4f}`"
                )
                lines.append(
                    f"- {model} precision: mean=`{payload['metrics']['precision']['mean']:.4f}` "
                    f"std=`{payload['metrics']['precision']['std']:.4f}`"
                )
                lines.append(
                    f"- {model} recall: mean=`{payload['metrics']['recall']['mean']:.4f}` "
                    f"std=`{payload['metrics']['recall']['std']:.4f}`"
                )
            lines.append("")

    lines.append("## Individual runs")
    lines.append("")
    for row in summary["runs"]:
        lines.append(
            f"- {row['model']} fold={row['de_chazal_val_fold']} seed={row['seed']} split_seed={row['split_seed']} "
            f"f1=`{row['f1']:.4f}` precision=`{row['precision']:.4f}` "
            f"recall=`{row['recall']:.4f}` output=`{row['output_dir']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_args = strip_remainder_prefix(args.train_args)

    rows: list[dict[str, object]] = []
    folds = args.folds if args.folds else [None]
    for model in args.models:
        for fold in folds:
            for seed in args.seeds:
                run_args = build_run_args(
                    train_args,
                    model=model,
                    seed=seed,
                    export_runs=args.export_runs,
                    fold=fold,
                )
                print(
                    f"Running model={model} fold={run_args.de_chazal_val_fold} "
                    f"seed={seed} split_policy={run_args.split_policy} split_seed={run_args.split_seed}"
                )
                output_dir, metrics = run_experiment(run_args)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "split_policy": run_args.split_policy,
                        "split_seed": run_args.split_seed,
                        "de_chazal_val_fold": int(run_args.de_chazal_val_fold),
                        "accuracy": float(metrics["accuracy"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "threshold": float(metrics["threshold"]),
                        "latency_ms_per_window_cpu": float(metrics["latency_ms_per_window_cpu"]),
                        "output_dir": str(output_dir.resolve()),
                    }
                )

    summary = summarize_runs(rows)
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    summary["models"] = args.models
    summary["seeds"] = args.seeds
    summary["folds"] = args.folds
    summary["train_args"] = train_args

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = args.summary_dir / f"{stamp}_summary.json"
    md_path = args.summary_dir / f"{stamp}_summary.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Sweep summary saved to: {json_path.resolve()}")
    print(f"Sweep report saved to: {md_path.resolve()}")
    return 0
