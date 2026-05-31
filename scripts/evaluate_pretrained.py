from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heart.data import build_dataset, make_record_splits, select_split, split_summary
from heart.models import build_model
from heart.train import (
    build_dataloaders,
    evaluate_with_threshold,
    get_device,
    benchmark_latency_ms,
    resolve_context_radius,
    resolve_history_beats,
    resolve_personalization_flags,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a released ECG checkpoint.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to a released .pt state_dict.")
    parser.add_argument("--metrics-json", type=Path, help="Optional metrics.json to reuse its saved threshold.")
    parser.add_argument("--threshold", type=float, help="Decision threshold. Overrides --metrics-json.")
    parser.add_argument(
        "--model",
        choices=["baseline", "attention", "morph", "context", "rr-context", "personalized-rr-context"],
        default="rr-context",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--leads", nargs="+", type=int, default=[0])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-policy", choices=["random-record", "de-chazal-interpatient"], default="de-chazal-interpatient")
    parser.add_argument("--de-chazal-val-mode", choices=["random", "fold", "beat-balanced-fold"], default="beat-balanced-fold")
    parser.add_argument("--de-chazal-num-folds", type=int, default=5)
    parser.add_argument("--de-chazal-val-fold", type=int, default=0)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--history-beats", type=int, default=8)
    parser.add_argument("--disable-personal-rr-baseline", action="store_true")
    parser.add_argument("--disable-history-prototype", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore processed cache and rebuild from PhysioNet raw data.")
    parser.add_argument("--benchmark-runs", type=int, default=300, help="Number of single-window CPU latency runs.")
    return parser


def resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.metrics_json is not None:
        with args.metrics_json.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "threshold" not in payload:
            raise ValueError(f"{args.metrics_json} does not contain a threshold field.")
        return float(payload["threshold"])
    return 0.5


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = get_device()
    threshold = resolve_threshold(args)

    context_radius = resolve_context_radius(args.model, args.context_radius)
    history_beats = resolve_history_beats(args.model, args.history_beats)
    use_personal_rr, use_history_prototype = resolve_personalization_flags(
        args.model,
        disable_personal_rr_baseline=args.disable_personal_rr_baseline,
        disable_history_prototype=args.disable_history_prototype,
    )

    x, y, record_ids, rr, x_rr = build_dataset(
        data_root=args.data_root,
        window_size=args.window_size,
        leads=args.leads,
        force_rebuild=args.force_rebuild,
    )
    splits = make_record_splits(
        record_ids=record_ids,
        y=y,
        seed=args.split_seed,
        policy=args.split_policy,
        de_chazal_val_mode=args.de_chazal_val_mode,
        de_chazal_num_folds=args.de_chazal_num_folds,
        de_chazal_val_fold=args.de_chazal_val_fold,
    )
    train_split = select_split(x, y, record_ids, splits["train"], rr=rr, x_rr=x_rr)
    val_split = select_split(x, y, record_ids, splits["val"], rr=rr, x_rr=x_rr)
    test_split = select_split(x, y, record_ids, splits["test"], rr=rr, x_rr=x_rr)

    print(split_summary("train", train_split))
    print(split_summary("val", val_split))
    print(split_summary("test", test_split))

    _, _, test_loader, _, example_input = build_dataloaders(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=0,
        sampler_mode="none",
        context_radius=context_radius,
        history_beats=history_beats,
    )

    model = build_model(
        args.model,
        dropout=args.dropout,
        in_channels=len(args.leads),
        context_beats=context_radius * 2 + 1,
        rr_feature_dim=0 if train_split.rr is None else int(train_split.rr.shape[1]),
        history_beats=history_beats,
        use_personal_rr=use_personal_rr,
        use_history_prototype=use_history_prototype,
    ).to(device)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    metrics = evaluate_with_threshold(model, test_loader, device=device, threshold=threshold)
    if args.benchmark_runs > 0:
        latency_ms = benchmark_latency_ms(model, example_input, runs=args.benchmark_runs)
        metrics["latency_ms_per_window_cpu"] = latency_ms
        metrics["windows_per_second_cpu"] = 1000.0 / latency_ms if latency_ms > 0 else None
        metrics["benchmark_runs"] = int(args.benchmark_runs)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
