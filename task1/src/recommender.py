#!/usr/bin/env python3
"""Task 1 rating prediction pipeline.

The script parses the course data format, evaluates several recommendation
models on a validation split, trains the selected model on all training data,
and writes predictions for test.txt in the required block format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Rating = Tuple[int, int, float]
TestBlock = Tuple[int, List[int]]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "作业说明" / "任务一" / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train.txt"
DEFAULT_TEST = DEFAULT_DATA_DIR / "test.txt"
DEFAULT_OUT = ROOT / "task1" / "results" / "predictions.txt"
DEFAULT_METRICS = ROOT / "task1" / "results" / "metrics.json"
DEFAULT_LOG = ROOT / "task1" / "results" / "experiment.log"


def clamp(value: float, low: float = 10.0, high: float = 100.0) -> float:
    return min(high, max(low, value))


def read_non_empty_line(handle) -> str | None:
    for line in handle:
        line = line.strip()
        if line:
            return line
    return None


def parse_header(line: str) -> Tuple[int, int]:
    if "|" not in line:
        raise ValueError(f"Expected header '<user>|<count>', got: {line!r}")
    user_text, count_text = line.split("|", 1)
    return int(user_text.strip()), int(count_text.strip())


def read_train(path: Path) -> List[Rating]:
    ratings: List[Rating] = []
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        while True:
            header = read_non_empty_line(handle)
            if header is None:
                break
            user, count = parse_header(header)
            for _ in range(count):
                line = read_non_empty_line(handle)
                if line is None:
                    raise ValueError(f"Unexpected EOF while reading user {user}")
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Expected '<item> <score>', got: {line!r}")
                ratings.append((user, int(parts[0]), float(parts[1])))
    return ratings


def read_test(path: Path) -> List[TestBlock]:
    blocks: List[TestBlock] = []
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        while True:
            header = read_non_empty_line(handle)
            if header is None:
                break
            user, count = parse_header(header)
            items: List[int] = []
            for _ in range(count):
                line = read_non_empty_line(handle)
                if line is None:
                    raise ValueError(f"Unexpected EOF while reading test user {user}")
                items.append(int(line.split()[0]))
            blocks.append((user, items))
    return blocks


def write_predictions(path: Path, blocks: Sequence[TestBlock], model: "Predictor") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user, items in blocks:
            handle.write(f"{user}|{len(items)}\n")
            for item in items:
                handle.write(f"{item} {model.predict(user, item):.3f}\n")


def validate_prediction_file(test_blocks: Sequence[TestBlock], result_path: Path) -> None:
    result_blocks = read_result(result_path)
    if len(test_blocks) != len(result_blocks):
        raise ValueError(
            f"Result block count mismatch: expected {len(test_blocks)}, got {len(result_blocks)}"
        )
    for idx, ((user, items), (out_user, out_items)) in enumerate(
        zip(test_blocks, result_blocks), start=1
    ):
        if user != out_user:
            raise ValueError(f"Block {idx}: expected user {user}, got {out_user}")
        if items != [item for item, _ in out_items]:
            raise ValueError(f"Block {idx}: item sequence does not match test.txt")


def read_result(path: Path) -> List[Tuple[int, List[Tuple[int, float]]]]:
    blocks: List[Tuple[int, List[Tuple[int, float]]]] = []
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        while True:
            header = read_non_empty_line(handle)
            if header is None:
                break
            user, count = parse_header(header)
            rows: List[Tuple[int, float]] = []
            for _ in range(count):
                line = read_non_empty_line(handle)
                if line is None:
                    raise ValueError(f"Unexpected EOF while reading result user {user}")
                item, score = line.split()[:2]
                rows.append((int(item), float(score)))
            blocks.append((user, rows))
    return blocks


@dataclass
class DataStats:
    train_users: int
    train_items: int
    train_ratings: int
    test_users: int
    test_items: int
    test_pairs: int
    cold_test_users: int
    cold_test_items: int
    rating_min: float
    rating_max: float
    rating_mean: float
    rating_median: float
    rating_stdev: float
    density: float
    train_ratings_per_user_min: int
    train_ratings_per_user_max: int
    train_ratings_per_user_mean: float
    test_pairs_per_user_min: int
    test_pairs_per_user_max: int
    test_pairs_per_user_mean: float
    score_distribution: Dict[str, int]


def describe_data(ratings: Sequence[Rating], test_blocks: Sequence[TestBlock]) -> DataStats:
    train_users = {user for user, _, _ in ratings}
    train_items = {item for _, item, _ in ratings}
    test_users = {user for user, _ in test_blocks}
    test_items = {item for _, items in test_blocks for item in items}
    scores = [score for _, _, score in ratings]
    user_counts = Counter(user for user, _, _ in ratings)
    test_counts = [len(items) for _, items in test_blocks]
    score_counts = Counter(int(score) if score.is_integer() else score for score in scores)

    return DataStats(
        train_users=len(train_users),
        train_items=len(train_items),
        train_ratings=len(ratings),
        test_users=len(test_users),
        test_items=len(test_items),
        test_pairs=sum(test_counts),
        cold_test_users=len(test_users - train_users),
        cold_test_items=len(test_items - train_items),
        rating_min=min(scores),
        rating_max=max(scores),
        rating_mean=sum(scores) / len(scores),
        rating_median=statistics.median(scores),
        rating_stdev=statistics.pstdev(scores),
        density=len(ratings) / (len(train_users) * len(train_items)),
        train_ratings_per_user_min=min(user_counts.values()),
        train_ratings_per_user_max=max(user_counts.values()),
        train_ratings_per_user_mean=len(ratings) / len(train_users),
        test_pairs_per_user_min=min(test_counts),
        test_pairs_per_user_max=max(test_counts),
        test_pairs_per_user_mean=sum(test_counts) / len(test_counts),
        score_distribution={str(k): score_counts[k] for k in sorted(score_counts)},
    )


def split_validation(
    ratings: Sequence[Rating], val_ratio: float, seed: int
) -> Tuple[List[Rating], List[Rating]]:
    rng = random.Random(seed)
    indices = list(range(len(ratings)))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:val_size])
    train_part: List[Rating] = []
    val_part: List[Rating] = []
    for idx, row in enumerate(ratings):
        if idx in val_indices:
            val_part.append(row)
        else:
            train_part.append(row)
    return train_part, val_part


class Predictor:
    name = "predictor"

    def fit(self, ratings: Sequence[Rating]) -> None:
        raise NotImplementedError

    def predict(self, user: int, item: int) -> float:
        raise NotImplementedError

    def parameter_bytes(self) -> int:
        return 0


class GlobalMeanModel(Predictor):
    name = "global_mean"

    def __init__(self) -> None:
        self.global_mean = 0.0

    def fit(self, ratings: Sequence[Rating]) -> None:
        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)

    def predict(self, user: int, item: int) -> float:
        return clamp(self.global_mean)


class ShrunkMeanModel(Predictor):
    name = "shrunk_user_item_mean"

    def __init__(self, user_alpha: float = 20.0, item_alpha: float = 40.0) -> None:
        self.user_alpha = user_alpha
        self.item_alpha = item_alpha
        self.global_mean = 0.0
        self.user_bias: Dict[int, float] = {}
        self.item_bias: Dict[int, float] = {}

    def fit(self, ratings: Sequence[Rating]) -> None:
        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)
        user_sum: Dict[int, float] = defaultdict(float)
        user_count: Dict[int, int] = defaultdict(int)
        item_sum: Dict[int, float] = defaultdict(float)
        item_count: Dict[int, int] = defaultdict(int)
        for user, item, score in ratings:
            user_sum[user] += score
            user_count[user] += 1
            item_sum[item] += score
            item_count[item] += 1

        self.user_bias = {
            user: (total - count * self.global_mean) / (count + self.user_alpha)
            for user, total in user_sum.items()
            for count in [user_count[user]]
        }
        self.item_bias = {
            item: (total - count * self.global_mean) / (count + self.item_alpha)
            for item, total in item_sum.items()
            for count in [item_count[item]]
        }

    def predict(self, user: int, item: int) -> float:
        return clamp(
            self.global_mean + self.user_bias.get(user, 0.0) + self.item_bias.get(item, 0.0)
        )

    def parameter_bytes(self) -> int:
        return 8 * (1 + len(self.user_bias) + len(self.item_bias))


class BiasSGDModel(Predictor):
    name = "regularized_bias_sgd"

    def __init__(
        self,
        epochs: int = 35,
        lr: float = 0.007,
        reg: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.global_mean = 0.0
        self.user_bias: Dict[int, float] = {}
        self.item_bias: Dict[int, float] = {}

    def fit(self, ratings: Sequence[Rating]) -> None:
        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)
        users = sorted({user for user, _, _ in ratings})
        items = sorted({item for _, item, _ in ratings})
        self.user_bias = {user: 0.0 for user in users}
        self.item_bias = {item: 0.0 for item in items}
        rows = list(ratings)
        rng = random.Random(self.seed)

        for _ in range(self.epochs):
            rng.shuffle(rows)
            for user, item, score in rows:
                pred = self.global_mean + self.user_bias[user] + self.item_bias[item]
                err = score - pred
                self.user_bias[user] += self.lr * (err - self.reg * self.user_bias[user])
                self.item_bias[item] += self.lr * (err - self.reg * self.item_bias[item])

    def predict(self, user: int, item: int) -> float:
        return clamp(
            self.global_mean + self.user_bias.get(user, 0.0) + self.item_bias.get(item, 0.0)
        )

    def parameter_bytes(self) -> int:
        return 8 * (1 + len(self.user_bias) + len(self.item_bias))


class MatrixFactorizationModel(Predictor):
    name = "biased_matrix_factorization"

    def __init__(
        self,
        factors: int = 24,
        epochs: int = 45,
        lr: float = 0.006,
        reg_bias: float = 0.03,
        reg_factor: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.seed = seed
        self.global_mean = 0.0
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.user_bias: List[float] = []
        self.item_bias: List[float] = []
        self.user_factors: List[List[float]] = []
        self.item_factors: List[List[float]] = []

    def fit(self, ratings: Sequence[Rating]) -> None:
        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)
        users = sorted({user for user, _, _ in ratings})
        items = sorted({item for _, item, _ in ratings})
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}

        rng = random.Random(self.seed)
        scale = 0.1 / math.sqrt(self.factors)
        self.user_bias = [0.0 for _ in users]
        self.item_bias = [0.0 for _ in items]
        self.user_factors = [
            [rng.uniform(-scale, scale) for _ in range(self.factors)] for _ in users
        ]
        self.item_factors = [
            [rng.uniform(-scale, scale) for _ in range(self.factors)] for _ in items
        ]

        indexed_rows = [
            (self.user_to_idx[user], self.item_to_idx[item], score)
            for user, item, score in ratings
        ]

        for epoch in range(self.epochs):
            rng.shuffle(indexed_rows)
            lr = self.lr / (1.0 + 0.03 * epoch)
            for user_idx, item_idx, score in indexed_rows:
                user_vec = self.user_factors[user_idx]
                item_vec = self.item_factors[item_idx]
                dot = 0.0
                for k in range(self.factors):
                    dot += user_vec[k] * item_vec[k]

                pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + dot
                err = score - pred

                self.user_bias[user_idx] += lr * (
                    err - self.reg_bias * self.user_bias[user_idx]
                )
                self.item_bias[item_idx] += lr * (
                    err - self.reg_bias * self.item_bias[item_idx]
                )

                for k in range(self.factors):
                    old_user = user_vec[k]
                    old_item = item_vec[k]
                    user_vec[k] += lr * (err * old_item - self.reg_factor * old_user)
                    item_vec[k] += lr * (err * old_user - self.reg_factor * old_item)

    def predict(self, user: int, item: int) -> float:
        user_idx = self.user_to_idx.get(user)
        item_idx = self.item_to_idx.get(item)
        pred = self.global_mean
        if user_idx is not None:
            pred += self.user_bias[user_idx]
        if item_idx is not None:
            pred += self.item_bias[item_idx]
        if user_idx is not None and item_idx is not None:
            user_vec = self.user_factors[user_idx]
            item_vec = self.item_factors[item_idx]
            for k in range(self.factors):
                pred += user_vec[k] * item_vec[k]
        return clamp(pred)

    def parameter_bytes(self) -> int:
        user_count = len(self.user_bias)
        item_count = len(self.item_bias)
        return 8 * (1 + user_count + item_count + self.factors * (user_count + item_count))


def rmse(model: Predictor, ratings: Sequence[Rating]) -> float:
    total = 0.0
    for user, item, score in ratings:
        err = score - model.predict(user, item)
        total += err * err
    return math.sqrt(total / len(ratings))


def mae(model: Predictor, ratings: Sequence[Rating]) -> float:
    return sum(abs(score - model.predict(user, item)) for user, item, score in ratings) / len(
        ratings
    )


def profile_fit(model: Predictor, ratings: Sequence[Rating]) -> Tuple[float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(ratings)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024 / 1024


def evaluate_models(
    train_part: Sequence[Rating],
    val_part: Sequence[Rating],
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
    seed: int,
    include_mf: bool,
) -> List[Dict[str, float | int | str]]:
    models: List[Predictor] = [
        GlobalMeanModel(),
        ShrunkMeanModel(),
        BiasSGDModel(epochs=epochs, lr=lr, reg=reg, seed=seed),
    ]
    if include_mf:
        models.append(MatrixFactorizationModel(factors=factors, epochs=epochs, seed=seed))
    rows: List[Dict[str, float | int | str]] = []
    for model in models:
        train_seconds, peak_mib = profile_fit(model, train_part)
        rows.append(
            {
                "model": model.name,
                "validation_rmse": rmse(model, val_part),
                "validation_mae": mae(model, val_part),
                "train_seconds": train_seconds,
                "peak_tracemalloc_mib": peak_mib,
                "parameter_bytes_estimate": model.parameter_bytes(),
            }
        )
    return rows


def make_final_model(
    model_name: str, factors: int, epochs: int, lr: float, reg: float, seed: int
) -> Predictor:
    if model_name == "global":
        return GlobalMeanModel()
    if model_name == "mean":
        return ShrunkMeanModel()
    if model_name == "bias":
        return BiasSGDModel(epochs=epochs, lr=lr, reg=reg, seed=seed)
    if model_name == "mf":
        return MatrixFactorizationModel(factors=factors, epochs=epochs, seed=seed)
    raise ValueError(f"Unknown model: {model_name}")


def asdict_dataclass(stats: DataStats) -> Dict[str, object]:
    return dict(stats.__dict__)


def format_log(metrics: Dict[str, object]) -> str:
    lines = [
        "Task 1 experiment log",
        f"train ratings: {metrics['data_stats']['train_ratings']}",
        f"test pairs: {metrics['data_stats']['test_pairs']}",
        "",
        "Validation results:",
    ]
    for row in metrics.get("validation", []):
        lines.append(
            "- {model}: RMSE={rmse:.4f}, MAE={mae:.4f}, train={seconds:.3f}s, peak={peak:.3f}MiB".format(
                model=row["model"],
                rmse=row["validation_rmse"],
                mae=row["validation_mae"],
                seconds=row["train_seconds"],
                peak=row["peak_tracemalloc_mib"],
            )
        )
    final = metrics["final_model"]
    lines.extend(
        [
            "",
            "Final model:",
            f"- model: {final['model']}",
            f"- train_seconds: {final['train_seconds']:.3f}",
            f"- predict_seconds: {final['predict_seconds']:.3f}",
            f"- peak_tracemalloc_mib: {final['peak_tracemalloc_mib']:.3f}",
            f"- output: {metrics['paths']['output']}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 recommendation pipeline")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--model", choices=["global", "mean", "bias", "mf"], default="bias")
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.007, help="Learning rate for bias SGD.")
    parser.add_argument("--reg", type=float, default=0.05, help="L2 regularization for bias SGD.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--include-mf-validation",
        action="store_true",
        help="Also evaluate matrix factorization on the validation split. This is slower.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path = args.train.resolve()
    test_path = args.test.resolve()
    output_path = args.out.resolve()
    metrics_path = args.metrics.resolve()
    log_path = args.log.resolve()

    ratings = read_train(train_path)
    test_blocks = read_test(test_path)
    data_stats = describe_data(ratings, test_blocks)

    validation_rows: List[Dict[str, float | int | str]] = []
    if not args.skip_validation and args.val_ratio > 0:
        train_part, val_part = split_validation(ratings, args.val_ratio, args.seed)
        validation_rows = evaluate_models(
            train_part=train_part,
            val_part=val_part,
            factors=args.factors,
            epochs=args.epochs,
            lr=args.lr,
            reg=args.reg,
            seed=args.seed,
            include_mf=args.include_mf_validation,
        )

    final_model = make_final_model(args.model, args.factors, args.epochs, args.lr, args.reg, args.seed)
    final_train_seconds, final_peak_mib = profile_fit(final_model, ratings)

    start = time.perf_counter()
    write_predictions(output_path, test_blocks, final_model)
    predict_seconds = time.perf_counter() - start
    validate_prediction_file(test_blocks, output_path)

    metrics: Dict[str, object] = {
        "paths": {
            "train": str(train_path),
            "test": str(test_path),
            "output": str(output_path),
        },
        "config": {
            "model": args.model,
            "factors": args.factors,
            "epochs": args.epochs,
            "lr": args.lr,
            "reg": args.reg,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "data_stats": asdict_dataclass(data_stats),
        "validation": validation_rows,
        "final_model": {
            "model": final_model.name,
            "train_seconds": final_train_seconds,
            "predict_seconds": predict_seconds,
            "peak_tracemalloc_mib": final_peak_mib,
            "parameter_bytes_estimate": final_model.parameter_bytes(),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(format_log(metrics), encoding="utf-8")

    print(format_log(metrics), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
