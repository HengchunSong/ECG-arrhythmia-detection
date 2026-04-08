from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    ECGBeatDataset,
    ECGContextDataset,
    ECGPersonalizedDataset,
    ECGRRContextDataset,
    build_dataset,
    make_record_splits,
    select_split,
    split_summary,
)
from .models import build_model


@dataclass
class TrainConfig:
    data_root: str
    model: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    train_ratio: float
    val_ratio: float
    window_size: int
    leads: list[int]
    seed: int
    split_seed: int
    num_workers: int
    split_policy: str
    de_chazal_val_mode: str
    de_chazal_num_folds: int
    de_chazal_val_fold: int
    context_radius: int
    history_beats: int
    personalized_use_rr_baseline: bool
    personalized_use_history_prototype: bool
    force_rebuild: bool
    records_limit: int | None
    max_beats_per_record: int | None
    sampler: str
    class_weight: str
    export: bool
    run_tag: str | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIT-BIH ventricular arrhythmia prototype")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--model",
        choices=["baseline", "attention", "morph", "context", "rr-context", "personalized-rr-context"],
        default="attention",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--leads", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--split-policy",
        choices=["random-record", "de-chazal-interpatient"],
        default="random-record",
    )
    parser.add_argument("--de-chazal-val-mode", choices=["random", "fold", "beat-balanced-fold"], default="random")
    parser.add_argument("--de-chazal-num-folds", type=int, default=5)
    parser.add_argument("--de-chazal-val-fold", type=int, default=0)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--history-beats", type=int, default=8)
    parser.add_argument("--disable-personal-rr-baseline", action="store_true")
    parser.add_argument("--disable-history-prototype", action="store_true")
    parser.add_argument("--run-tag", type=str)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--records-limit", type=int)
    parser.add_argument("--max-beats-per-record", type=int)
    parser.add_argument("--sampler", choices=["weighted", "none"], default="weighted")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--no-export", dest="export", action="store_false")
    parser.set_defaults(export=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_context_radius(model_name: str, requested_radius: int) -> int:
    if model_name.lower() in {"context", "rr-context", "personalized-rr-context"}:
        if requested_radius < 1:
            raise ValueError("context model requires --context-radius >= 1.")
        return requested_radius
    return 0


def resolve_history_beats(model_name: str, requested_history_beats: int) -> int:
    if model_name.lower() == "personalized-rr-context":
        if requested_history_beats < 1:
            raise ValueError("personalized-rr-context requires --history-beats >= 1.")
        return requested_history_beats
    return 0


def resolve_personalization_flags(
    model_name: str,
    disable_personal_rr_baseline: bool,
    disable_history_prototype: bool,
) -> tuple[bool, bool]:
    if model_name.lower() != "personalized-rr-context":
        return False, False
    return (not disable_personal_rr_baseline), (not disable_history_prototype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to_device(batch: object, device: torch.device) -> object:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def call_model(model: nn.Module, batch: object) -> torch.Tensor:
    if torch.is_tensor(batch):
        return model(batch)
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def add_batch_dim(batch: object) -> object:
    if torch.is_tensor(batch):
        return batch.unsqueeze(0)
    if isinstance(batch, dict):
        return {key: add_batch_dim(value) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(add_batch_dim(value) for value in batch)
    if isinstance(batch, list):
        return [add_batch_dim(value) for value in batch]
    return batch


def describe_input_structure(batch: object) -> object:
    if torch.is_tensor(batch):
        return list(batch.shape)
    if isinstance(batch, dict):
        return {key: describe_input_structure(value) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return [describe_input_structure(value) for value in batch]
    if isinstance(batch, list):
        return [describe_input_structure(value) for value in batch]
    return str(type(batch))


def build_dataloaders(
    train_split,
    val_split,
    test_split,
    model_name: str,
    batch_size: int,
    num_workers: int,
    sampler_mode: str,
    context_radius: int = 0,
    history_beats: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, object]:
    if model_name == "personalized-rr-context":
        if train_split.rr is None or train_split.x_rr is None:
            raise ValueError("Personalized RR-aware model requires rr and x_rr arrays in the split data.")
        train_ds = ECGPersonalizedDataset(
            train_split.x,
            train_split.y,
            train_split.records,
            rr=train_split.rr,
            x_rr=train_split.x_rr,
            context_radius=context_radius,
            history_beats=history_beats,
            augment=True,
        )
        val_ds = ECGPersonalizedDataset(
            val_split.x,
            val_split.y,
            val_split.records,
            rr=val_split.rr,
            x_rr=val_split.x_rr,
            context_radius=context_radius,
            history_beats=history_beats,
            augment=False,
        )
        test_ds = ECGPersonalizedDataset(
            test_split.x,
            test_split.y,
            test_split.records,
            rr=test_split.rr,
            x_rr=test_split.x_rr,
            context_radius=context_radius,
            history_beats=history_beats,
            augment=False,
        )
    elif model_name == "rr-context":
        if train_split.rr is None or train_split.x_rr is None:
            raise ValueError("RR-aware model requires rr and x_rr arrays in the split data.")
        train_ds = ECGRRContextDataset(
            train_split.x,
            train_split.y,
            train_split.records,
            rr=train_split.rr,
            x_rr=train_split.x_rr,
            context_radius=context_radius,
            augment=True,
        )
        val_ds = ECGRRContextDataset(
            val_split.x,
            val_split.y,
            val_split.records,
            rr=val_split.rr,
            x_rr=val_split.x_rr,
            context_radius=context_radius,
            augment=False,
        )
        test_ds = ECGRRContextDataset(
            test_split.x,
            test_split.y,
            test_split.records,
            rr=test_split.rr,
            x_rr=test_split.x_rr,
            context_radius=context_radius,
            augment=False,
        )
    elif context_radius > 0:
        train_ds = ECGContextDataset(
            train_split.x,
            train_split.y,
            train_split.records,
            context_radius=context_radius,
            augment=True,
        )
        val_ds = ECGContextDataset(
            val_split.x,
            val_split.y,
            val_split.records,
            context_radius=context_radius,
            augment=False,
        )
        test_ds = ECGContextDataset(
            test_split.x,
            test_split.y,
            test_split.records,
            context_radius=context_radius,
            augment=False,
        )
    else:
        train_ds = ECGBeatDataset(train_split.x, train_split.y, augment=True)
        val_ds = ECGBeatDataset(val_split.x, val_split.y, augment=False)
        test_ds = ECGBeatDataset(test_split.x, test_split.y, augment=False)

    class_counts = np.bincount(train_split.y, minlength=2)
    class_weights = (class_counts.sum() / np.clip(class_counts, a_min=1, a_max=None)).astype(np.float32)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if sampler_mode == "weighted":
        sample_weights = class_weights[train_split.y]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    example_input, _ = train_ds[0]
    return train_loader, val_loader, test_loader, class_weights, example_input


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for xb, yb in loader:
        xb = move_to_device(xb, device)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = call_model(model, xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    if not all_targets:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    loss_value = total_loss / max(len(loader.dataset), 1)
    return {
        "loss": float(loss_value),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_with_confusion(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    return evaluate_with_threshold(model, loader, device=device, threshold=0.5)


def collect_targets_and_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb in loader:
            logits = call_model(model, move_to_device(xb, device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_targets.append(yb.numpy())

    if not all_targets:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, object]:
    if y_true.size == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_pred = (y_prob >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, object]]:
    if y_true.size == 0:
        return 0.5, compute_binary_metrics(y_true, y_prob, threshold=0.5)

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics["f1"] if metrics["f1"] is not None else -1.0
        best_score = best_metrics["f1"] if best_metrics["f1"] is not None else -1.0
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def evaluate_with_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    y_true, y_prob = collect_targets_and_probs(model, loader, device=device)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate_tuned_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, object]]:
    y_true, y_prob = collect_targets_and_probs(model, loader, device=device)
    threshold, metrics = tune_threshold(y_true, y_prob)
    metrics["threshold"] = float(threshold)
    return threshold, metrics


def benchmark_latency_ms(model: nn.Module, example_input: object, runs: int = 300) -> float:
    model_cpu = copy.deepcopy(model).cpu().eval()
    example = add_batch_dim(example_input)
    with torch.inference_mode():
        for _ in range(40):
            _ = call_model(model_cpu, example)
        start = time.perf_counter()
        for _ in range(runs):
            _ = call_model(model_cpu, example)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / runs)


def export_artifacts(model: nn.Module, output_dir: Path, example_input: object) -> None:
    model_cpu = copy.deepcopy(model).cpu().eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(output_dir / "best.ts"))

    try:
        quantized = torch.quantization.quantize_dynamic(
            copy.deepcopy(model_cpu),
            {nn.Linear, nn.GRU},
            dtype=torch.qint8,
        )
        torch.save(
            {
                "model_class": model_cpu.__class__.__name__,
                "input_shape": describe_input_structure(example_input),
                "state_dict": quantized.state_dict(),
            },
            output_dir / "best_dynamic_q.pt",
        )
    except Exception as exc:
        warning_path = output_dir / "export_warning.txt"
        warning_path.write_text(
            f"Dynamic quantized export skipped: {exc}\n",
            encoding="utf-8",
        )


def sanitize_run_tag(run_tag: str | None) -> str | None:
    if run_tag is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_tag.strip().lower())
    safe = safe.strip("-_")
    return safe or None


def make_output_dir(model_name: str, run_tag: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_tag = sanitize_run_tag(run_tag)
    suffix = f"_{safe_tag}" if safe_tag else ""
    path = Path("artifacts") / f"{stamp}_{model_name}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_experiment(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    seed_everything(args.seed)
    device = get_device()

    if args.split_policy == "de-chazal-interpatient" and args.records_limit:
        raise ValueError("records_limit is not supported with de-chazal-interpatient split.")

    context_radius = resolve_context_radius(args.model, args.context_radius)
    history_beats = resolve_history_beats(args.model, args.history_beats)
    use_personal_rr, use_history_prototype = resolve_personalization_flags(
        args.model,
        disable_personal_rr_baseline=args.disable_personal_rr_baseline,
        disable_history_prototype=args.disable_history_prototype,
    )

    config = TrainConfig(
        data_root=str(args.data_root),
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        window_size=args.window_size,
        leads=args.leads,
        seed=args.seed,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        split_policy=args.split_policy,
        de_chazal_val_mode=args.de_chazal_val_mode,
        de_chazal_num_folds=args.de_chazal_num_folds,
        de_chazal_val_fold=args.de_chazal_val_fold,
        context_radius=args.context_radius,
        history_beats=args.history_beats,
        personalized_use_rr_baseline=use_personal_rr,
        personalized_use_history_prototype=use_history_prototype,
        force_rebuild=args.force_rebuild,
        records_limit=args.records_limit,
        max_beats_per_record=args.max_beats_per_record,
        sampler=args.sampler,
        class_weight=args.class_weight,
        export=args.export,
        run_tag=args.run_tag,
    )

    x, y, record_ids, rr, x_rr = build_dataset(
        data_root=args.data_root,
        window_size=args.window_size,
        leads=args.leads,
        force_rebuild=args.force_rebuild,
        records_limit=args.records_limit,
        max_beats_per_record=args.max_beats_per_record,
    )
    splits = make_record_splits(
        record_ids=record_ids,
        y=y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
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

    train_loader, val_loader, test_loader, class_weights, example_input = build_dataloaders(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_mode=args.sampler,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_weight = None
    if args.class_weight == "balanced":
        loss_weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    output_dir = make_output_dir(args.model, run_tag=args.run_tag)
    with (output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device=device, optimizer=None)
        tuned_threshold_epoch, tuned_val_metrics = evaluate_tuned_validation(model, val_loader, device=device)
        scheduler.step(tuned_val_metrics["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_tuned_f1": tuned_val_metrics["f1"],
            "val_tuned_precision": tuned_val_metrics["precision"],
            "val_tuned_recall": tuned_val_metrics["recall"],
            "val_tuned_threshold": tuned_threshold_epoch,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f} "
            f"val_tuned_f1={tuned_val_metrics['f1']:.4f} val_tuned_thr={tuned_threshold_epoch:.2f}"
        )

        if tuned_val_metrics["f1"] > best_val_f1:
            best_val_f1 = tuned_val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best.pt")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    val_true, val_prob = collect_targets_and_probs(model, val_loader, device=device)
    best_threshold, tuned_val_metrics = tune_threshold(val_true, val_prob)
    final_metrics = evaluate_with_threshold(model, test_loader, device=device, threshold=best_threshold)
    final_metrics["validation_threshold_metrics"] = tuned_val_metrics
    final_metrics["latency_ms_per_window_cpu"] = benchmark_latency_ms(model, example_input)
    final_metrics["history"] = history
    final_metrics["splits"] = {
        "train": splits["train"].tolist(),
        "val": splits["val"].tolist(),
        "test": splits["test"].tolist(),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2)

    if args.export:
        export_artifacts(
            model,
            output_dir=output_dir,
            example_input=example_input,
        )

    return output_dir, final_metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir, final_metrics = run_experiment(args)
    print("\nTest metrics")
    print(json.dumps(final_metrics, indent=2))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")
    return 0
