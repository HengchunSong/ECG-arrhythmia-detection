from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset
import wfdb


MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

DE_CHAZAL_DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

DE_CHAZAL_DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
VENTRICULAR_SYMBOLS = {"V", "E"}


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    records: np.ndarray
    rr: np.ndarray | None = None
    x_rr: np.ndarray | None = None


class ECGBeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment
        self.sample_shape = tuple(self.x.shape[1:])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        beat = self.x[index].clone()
        if self.augment:
            beat = augment_beat(beat)
        return beat, self.y[index]


class ECGContextDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        records: np.ndarray,
        context_radius: int,
        augment: bool = False,
    ) -> None:
        if context_radius < 1:
            raise ValueError("context_radius must be at least 1 for context datasets.")

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.records = np.asarray(records)
        self.context_radius = context_radius
        self.augment = augment
        self.sample_shape = (context_radius * 2 + 1,) + tuple(self.x.shape[1:])
        self.context_indices = torch.from_numpy(
            build_context_index_map(self.records, context_radius=context_radius)
        ).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.x[self.context_indices[index]].clone()
        if self.augment:
            context = torch.stack([augment_beat(beat) for beat in context], dim=0)
        return context, self.y[index]


class ECGRRContextDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        records: np.ndarray,
        rr: np.ndarray,
        x_rr: np.ndarray,
        context_radius: int,
        augment: bool = False,
    ) -> None:
        if context_radius < 1:
            raise ValueError("context_radius must be at least 1 for RR-aware context datasets.")
        if rr.shape[0] != x.shape[0] or x_rr.shape[0] != x.shape[0]:
            raise ValueError("RR-aware arrays must align with the beat tensor.")

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.records = np.asarray(records)
        self.rr = torch.from_numpy(rr).float()
        self.x_rr = torch.from_numpy(x_rr).float()
        self.context_radius = context_radius
        self.augment = augment
        self.sample_shape = {
            "context": (context_radius * 2 + 1,) + tuple(self.x.shape[1:]),
            "rr_features": (context_radius * 2 + 1, self.rr.shape[1]),
            "normalized_center": tuple(self.x_rr.shape[1:]),
        }
        self.context_indices = torch.from_numpy(
            build_context_index_map(self.records, context_radius=context_radius)
        ).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        indices = self.context_indices[index]
        context = self.x[indices].clone()
        normalized_center = self.x_rr[index].clone()
        if self.augment:
            context = torch.stack([augment_beat(beat) for beat in context], dim=0)
            normalized_center = augment_beat(normalized_center)
        return {
            "context": context,
            "rr_features": self.rr[indices].clone(),
            "normalized_center": normalized_center,
        }, self.y[index]


class ECGPersonalizedDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        records: np.ndarray,
        rr: np.ndarray,
        x_rr: np.ndarray,
        context_radius: int,
        history_beats: int,
        augment: bool = False,
    ) -> None:
        if context_radius < 1:
            raise ValueError("context_radius must be at least 1 for personalized datasets.")
        if history_beats < 1:
            raise ValueError("history_beats must be at least 1 for personalized datasets.")
        if rr.shape[0] != x.shape[0] or x_rr.shape[0] != x.shape[0]:
            raise ValueError("Personalized arrays must align with the beat tensor.")

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.records = np.asarray(records)
        self.rr = torch.from_numpy(rr).float()
        self.x_rr = torch.from_numpy(x_rr).float()
        self.context_radius = context_radius
        self.history_beats = history_beats
        self.augment = augment

        self.context_indices = torch.from_numpy(
            build_context_index_map(self.records, context_radius=context_radius)
        ).long()
        history_index_map, history_mask = build_history_index_map(self.records, history_beats=history_beats)
        rr_baseline = compute_history_rr_baseline(rr, history_index_map, history_mask)

        self.history_indices = torch.from_numpy(history_index_map).long()
        self.history_mask = torch.from_numpy(history_mask).float()
        self.rr_baseline = torch.from_numpy(rr_baseline).float()
        self.sample_shape = {
            "context": (context_radius * 2 + 1,) + tuple(self.x.shape[1:]),
            "rr_features": (context_radius * 2 + 1, self.rr.shape[1]),
            "history_beats": (history_beats,) + tuple(self.x.shape[1:]),
            "history_mask": (history_beats,),
            "history_rr": (history_beats, self.rr.shape[1]),
            "rr_baseline": (self.rr_baseline.shape[1],),
            "normalized_center": tuple(self.x_rr.shape[1:]),
        }

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        context_indices = self.context_indices[index]
        history_indices = self.history_indices[index]
        history_mask = self.history_mask[index].clone()
        context = self.x[context_indices].clone()
        rr_features = self.rr[context_indices].clone()
        history_beats = self.x[history_indices].clone()
        history_rr = self.rr[history_indices].clone()
        normalized_center = self.x_rr[index].clone()
        if self.augment:
            context = torch.stack([augment_beat(beat) for beat in context], dim=0)
            history_beats = torch.stack([augment_beat(beat) for beat in history_beats], dim=0)
            normalized_center = augment_beat(normalized_center)

        history_beats = history_beats * history_mask.view(-1, 1, 1)
        history_rr = history_rr * history_mask.view(-1, 1)
        return {
            "context": context,
            "rr_features": rr_features,
            "history_beats": history_beats,
            "history_mask": history_mask,
            "history_rr": history_rr,
            "rr_baseline": self.rr_baseline[index].clone(),
            "normalized_center": normalized_center,
        }, self.y[index]


def augment_beat(beat: torch.Tensor) -> torch.Tensor:
    scale = torch.empty(1).uniform_(0.9, 1.1).item()
    noise = torch.randn_like(beat) * 0.01
    shift = int(torch.randint(-6, 7, (1,)).item())
    return torch.roll(beat * scale + noise, shifts=shift, dims=-1)


def build_context_index_map(records: np.ndarray, context_radius: int) -> np.ndarray:
    offsets = np.arange(-context_radius, context_radius + 1, dtype=np.int64)
    total = len(records)
    index_map = np.zeros((total, offsets.shape[0]), dtype=np.int64)

    start = 0
    while start < total:
        end = start + 1
        while end < total and records[end] == records[start]:
            end += 1

        local_positions = np.arange(end - start, dtype=np.int64)
        for pos in local_positions:
            neighbors = np.clip(pos + offsets, 0, local_positions.shape[0] - 1)
            index_map[start + pos] = start + neighbors
        start = end

    return index_map


def build_history_index_map(records: np.ndarray, history_beats: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(history_beats, 0, -1, dtype=np.int64)
    total = len(records)
    index_map = np.zeros((total, offsets.shape[0]), dtype=np.int64)
    mask = np.zeros((total, offsets.shape[0]), dtype=np.float32)

    start = 0
    while start < total:
        end = start + 1
        while end < total and records[end] == records[start]:
            end += 1

        local_positions = np.arange(end - start, dtype=np.int64)
        for pos in local_positions:
            history_positions = pos - offsets
            valid = history_positions >= 0
            clipped = np.clip(history_positions, 0, local_positions.shape[0] - 1)
            index_map[start + pos] = start + clipped
            mask[start + pos] = valid.astype(np.float32)
        start = end

    return index_map, mask


def compute_history_rr_baseline(
    rr: np.ndarray,
    history_index_map: np.ndarray,
    history_mask: np.ndarray,
) -> np.ndarray:
    rr_dim = rr.shape[1]
    stats = np.zeros((rr.shape[0], rr_dim * 2 + 1), dtype=np.float32)

    for index in range(rr.shape[0]):
        valid = history_mask[index] > 0.0
        if np.any(valid):
            history_rr = rr[history_index_map[index][valid]]
            mean = history_rr.mean(axis=0)
            std = history_rr.std(axis=0)
            count_fraction = float(valid.sum()) / float(history_mask.shape[1])
            stats[index] = np.concatenate(
                [
                    mean.astype(np.float32),
                    std.astype(np.float32),
                    np.asarray([count_fraction], dtype=np.float32),
                ],
                axis=0,
            )
    return stats


def ensure_even(window_size: int) -> int:
    if window_size % 2 != 0:
        raise ValueError("window_size must be even so beats stay centered.")
    return window_size


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 40.0) -> np.ndarray:
    nyquist = fs * 0.5
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    b, a = butter(3, [low, high], btype="bandpass")
    return filtfilt(b, a, signal).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = float(signal.mean())
    std = float(signal.std())
    return ((signal - mean) / (std + 1e-6)).astype(np.float32)


def resample_waveform(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[-1] == target_length:
        return signal.astype(np.float32)
    if signal.shape[-1] < 2:
        return np.repeat(signal.astype(np.float32), target_length, axis=-1)[..., :target_length]

    source_positions = np.linspace(0.0, 1.0, signal.shape[-1], dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    resampled = np.stack(
        [
            np.interp(target_positions, source_positions, channel).astype(np.float32)
            for channel in signal
        ],
        axis=0,
    )
    return resampled


def prepare_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    filtered = bandpass_filter(signal.astype(np.float32), fs=fs)
    return normalize_signal(filtered)


def download_mitdb(raw_root: Path, records: Iterable[str]) -> Path:
    mitdb_dir = raw_root / "mitdb"
    mitdb_dir.mkdir(parents=True, exist_ok=True)
    required = [mitdb_dir / f"{record}.atr" for record in records]
    if all(path.exists() for path in required):
        return mitdb_dir

    wfdb.dl_database("mitdb", dl_dir=str(mitdb_dir), records=list(records))
    return mitdb_dir


def extract_beats_from_record(
    record_name: str,
    raw_dir: Path,
    window_size: int,
    leads: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_path = raw_dir / record_name
    record = wfdb.rdrecord(str(record_path), channels=leads)
    annotation = wfdb.rdann(str(record_path), "atr")

    signals = np.stack(
        [prepare_signal(record.p_signal[:, idx], fs=record.fs) for idx in range(record.p_signal.shape[1])],
        axis=0,
    )
    half = ensure_even(window_size) // 2

    beat_samples: list[int] = []
    beat_labels: list[int] = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in NORMAL_SYMBOLS:
            label = 0
        elif symbol in VENTRICULAR_SYMBOLS:
            label = 1
        else:
            continue

        beat_samples.append(int(sample))
        beat_labels.append(label)

    if not beat_samples:
        empty_x = np.zeros((0, len(leads), window_size), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        empty_r = np.zeros((0,), dtype="<U3")
        empty_rr = np.zeros((0, 5), dtype=np.float32)
        empty_x_rr = np.zeros((0, len(leads), window_size), dtype=np.float32)
        return empty_x, empty_y, empty_r, empty_rr, empty_x_rr

    beat_samples_np = np.asarray(beat_samples, dtype=np.int64)
    beat_labels_np = np.asarray(beat_labels, dtype=np.int64)
    rr_intervals = np.diff(beat_samples_np).astype(np.float32)
    median_rr = float(np.median(rr_intervals)) if rr_intervals.size else float(window_size)
    median_rr = max(median_rr, 1.0)

    beats: list[np.ndarray] = []
    labels: list[int] = []
    records: list[str] = []
    rr_features: list[np.ndarray] = []
    normalized_beats: list[np.ndarray] = []

    for index, sample in enumerate(beat_samples_np):
        start = int(sample) - half
        end = int(sample) + half
        if start < 0 or end > signals.shape[1]:
            continue

        beat = signals[:, start:end]
        beat = np.stack([normalize_signal(channel) for channel in beat], axis=0)

        prev_rr = float(beat_samples_np[index] - beat_samples_np[index - 1]) if index > 0 else median_rr
        next_rr = (
            float(beat_samples_np[index + 1] - beat_samples_np[index])
            if index + 1 < beat_samples_np.shape[0] else median_rr
        )
        prev_rr = max(prev_rr, 1.0)
        next_rr = max(next_rr, 1.0)
        avg_rr = 0.5 * (prev_rr + next_rr)

        cycle_start = max(0, int(round(sample - prev_rr * 0.5)))
        cycle_end = min(signals.shape[1], int(round(sample + next_rr * 0.5)))
        cycle = signals[:, cycle_start:cycle_end]
        if cycle.shape[1] < 8:
            cycle = beat
        cycle = resample_waveform(cycle, target_length=window_size)
        cycle = np.stack([normalize_signal(channel) for channel in cycle], axis=0)

        beats.append(beat)
        labels.append(int(beat_labels_np[index]))
        records.append(record_name)
        rr_features.append(
            np.asarray(
                [
                    prev_rr / median_rr,
                    next_rr / median_rr,
                    avg_rr / median_rr,
                    (next_rr - prev_rr) / median_rr,
                    min(240.0, 60.0 * float(record.fs) / avg_rr) / 100.0,
                ],
                dtype=np.float32,
            )
        )
        normalized_beats.append(cycle)

    if not beats:
        empty_x = np.zeros((0, len(leads), window_size), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        empty_r = np.zeros((0,), dtype="<U3")
        empty_rr = np.zeros((0, 5), dtype=np.float32)
        empty_x_rr = np.zeros((0, len(leads), window_size), dtype=np.float32)
        return empty_x, empty_y, empty_r, empty_rr, empty_x_rr

    x = np.stack(beats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    r = np.asarray(records)
    rr = np.stack(rr_features).astype(np.float32)
    x_rr = np.stack(normalized_beats).astype(np.float32)
    return x, y, r, rr, x_rr


def build_dataset(
    data_root: Path,
    window_size: int = 256,
    leads: list[int] | None = None,
    force_rebuild: bool = False,
    records_limit: int | None = None,
    max_beats_per_record: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_leads = leads or [0]
    records = MITBIH_RECORDS[:records_limit] if records_limit else MITBIH_RECORDS
    raw_dir = download_mitdb(data_root / "raw", records=records)

    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    lead_suffix = "-".join(str(item) for item in selected_leads)
    suffix = f"w{window_size}_leads{lead_suffix}_n{len(records)}"
    if max_beats_per_record:
        suffix += f"_cap{max_beats_per_record}"
    cache_path = processed_dir / f"mitdb_binary_v2_{suffix}.npz"

    if cache_path.exists() and not force_rebuild:
        payload = np.load(cache_path, allow_pickle=False)
        return payload["x"], payload["y"], payload["records"], payload["rr"], payload["x_rr"]

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_records: list[np.ndarray] = []
    all_rr: list[np.ndarray] = []
    all_x_rr: list[np.ndarray] = []

    for record_name in records:
        x, y, r, rr, x_rr = extract_beats_from_record(
            record_name,
            raw_dir=raw_dir,
            window_size=window_size,
            leads=selected_leads,
        )
        if max_beats_per_record and x.shape[0] > max_beats_per_record:
            idx = np.linspace(0, x.shape[0] - 1, max_beats_per_record, dtype=np.int64)
            x = x[idx]
            y = y[idx]
            r = r[idx]
            rr = rr[idx]
            x_rr = x_rr[idx]
        all_x.append(x)
        all_y.append(y)
        all_records.append(r)
        all_rr.append(rr)
        all_x_rr.append(x_rr)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    record_ids = np.concatenate(all_records, axis=0)
    rr = np.concatenate(all_rr, axis=0)
    x_rr = np.concatenate(all_x_rr, axis=0)

    np.savez_compressed(cache_path, x=x, y=y, records=record_ids, rr=rr, x_rr=x_rr)
    return x, y, record_ids, rr, x_rr


def _resolve_test_count(num_records: int, test_size: float) -> int:
    count = int(round(num_records * test_size))
    count = max(1, count)
    count = min(num_records - 1, count)
    return count


def _split_records(records: np.ndarray, strata: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(records) <= 1:
        return records, np.asarray([], dtype=records.dtype)

    unique_strata = np.unique(strata)
    stratify = strata if unique_strata.shape[0] > 1 and np.min(np.bincount(strata)) >= 2 else None
    test_count = _resolve_test_count(len(records), test_size)
    return train_test_split(records, test_size=test_count, random_state=seed, stratify=stratify)


def make_record_splits(
    record_ids: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    policy: str = "random-record",
    de_chazal_val_mode: str = "random",
    de_chazal_num_folds: int = 5,
    de_chazal_val_fold: int = 0,
) -> dict[str, np.ndarray]:
    if policy == "de-chazal-interpatient":
        return make_de_chazal_splits(
            record_ids=record_ids,
            y=y,
            val_ratio=val_ratio,
            seed=seed,
            val_mode=de_chazal_val_mode,
            num_folds=de_chazal_num_folds,
            val_fold=de_chazal_val_fold,
            record_ids_full=record_ids,
        )

    if policy != "random-record":
        raise ValueError(f"Unknown split policy: {policy}")

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave room for a test split.")

    unique_records = np.array(sorted({str(item) for item in record_ids}))
    record_has_ventricular = np.asarray([
        int(np.any(y[record_ids == record] == 1))
        for record in unique_records
    ], dtype=np.int64)

    train_records, temp_records = _split_records(
        unique_records,
        record_has_ventricular,
        test_size=(1.0 - train_ratio),
        seed=seed,
    )

    temp_strata = np.asarray([
        int(np.any(y[record_ids == record] == 1))
        for record in temp_records
    ], dtype=np.int64)
    val_portion = val_ratio / (1.0 - train_ratio)
    if len(temp_records) <= 1:
        return {"train": train_records, "val": temp_records, "test": np.asarray([], dtype=temp_records.dtype)}
    val_records, test_records = _split_records(
        temp_records,
        temp_strata,
        test_size=(1.0 - val_portion),
        seed=seed,
    )

    return {"train": train_records, "val": val_records, "test": test_records}


def make_de_chazal_splits(
    record_ids: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
    val_mode: str = "random",
    num_folds: int = 5,
    val_fold: int = 0,
    record_ids_full: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    available_records = {str(item) for item in record_ids}
    ds1_records = np.asarray([record for record in DE_CHAZAL_DS1 if record in available_records])
    ds2_records = np.asarray([record for record in DE_CHAZAL_DS2 if record in available_records])

    if ds1_records.size == 0 or ds2_records.size == 0:
        raise ValueError("de-chazal-interpatient split requires records from the official DS1/DS2 lists.")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("For de-chazal-interpatient split, val_ratio must be between 0 and 1.")

    ds1_strata = np.asarray([
        int(np.any(y[record_ids == record] == 1))
        for record in ds1_records
    ], dtype=np.int64)

    if val_mode == "random":
        train_records, val_records = _split_records(
            ds1_records,
            ds1_strata,
            test_size=val_ratio,
            seed=seed,
        )
    elif val_mode == "fold":
        train_records, val_records = make_stratified_record_fold(
            records=ds1_records,
            strata=ds1_strata,
            num_folds=num_folds,
            fold_index=val_fold,
            seed=seed,
        )
    elif val_mode == "beat-balanced-fold":
        train_records, val_records = make_beat_balanced_record_fold(
            records=ds1_records,
            record_ids=record_ids if record_ids_full is None else record_ids_full,
            y=y,
            num_folds=num_folds,
            fold_index=val_fold,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown de Chazal validation mode: {val_mode}")

    return {"train": train_records, "val": val_records, "test": ds2_records}


def make_stratified_record_fold(
    records: np.ndarray,
    strata: np.ndarray,
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    class_counts = np.bincount(strata, minlength=2)
    max_folds = int(class_counts[class_counts > 0].min())
    if num_folds > max_folds:
        raise ValueError(
            f"num_folds={num_folds} is too large for the available DS1 strata; max supported is {max_folds}."
        )
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("fold_index must be in [0, num_folds).")

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_pairs = list(splitter.split(records, strata))
    train_idx, val_idx = fold_pairs[fold_index]
    return records[train_idx], records[val_idx]


def make_beat_balanced_record_fold(
    records: np.ndarray,
    record_ids: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("fold_index must be in [0, num_folds).")
    if num_folds > len(records):
        raise ValueError("num_folds cannot exceed the number of DS1 records.")

    positives = np.asarray([int(np.sum(y[record_ids == record] == 1)) for record in records], dtype=np.int64)
    totals = np.asarray([int(np.sum(record_ids == record)) for record in records], dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(records))
    sorted_indices = sorted(
        shuffled_indices.tolist(),
        key=lambda idx: (-positives[idx], -totals[idx], str(records[idx])),
    )

    fold_records: list[list[str]] = [[] for _ in range(num_folds)]
    fold_pos = np.zeros(num_folds, dtype=np.int64)
    fold_total = np.zeros(num_folds, dtype=np.int64)

    for idx in sorted_indices:
        target_fold = min(
            range(num_folds),
            key=lambda fold: (fold_pos[fold], fold_total[fold], len(fold_records[fold])),
        )
        fold_records[target_fold].append(str(records[idx]))
        fold_pos[target_fold] += positives[idx]
        fold_total[target_fold] += totals[idx]

    val_records = np.asarray(sorted(fold_records[fold_index]))
    train_records = np.asarray(sorted([
        record
        for fold, items in enumerate(fold_records)
        if fold != fold_index
        for record in items
    ]))
    return train_records, val_records


def select_split(
    x: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    split_records: np.ndarray,
    rr: np.ndarray | None = None,
    x_rr: np.ndarray | None = None,
) -> SplitData:
    mask = np.isin(record_ids, split_records)
    split_rr = rr[mask] if rr is not None else None
    split_x_rr = x_rr[mask] if x_rr is not None else None
    return SplitData(x=x[mask], y=y[mask], records=record_ids[mask], rr=split_rr, x_rr=split_x_rr)


def split_summary(split_name: str, split: SplitData) -> str:
    total = int(split.y.shape[0])
    positives = int((split.y == 1).sum())
    negatives = total - positives
    unique_records = len(set(split.records.tolist()))
    pos_ratio = positives / total if total else 0.0
    return (
        f"{split_name}: samples={total}, normal={negatives}, ventricular={positives}, "
        f"pos_ratio={pos_ratio:.4f}, records={unique_records}"
    )
