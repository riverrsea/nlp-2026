from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import DEFAULT_CONFIG_PATH, LABEL2ID, load_config
from utils import (
    clean_text,
    ensure_directories,
    extract_label_from_dirname,
    find_text_files,
    read_text_file,
    save_class_distribution_chart,
    save_json,
    save_length_distribution_chart,
    set_seed,
    setup_logging,
    summarize_numeric_series,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 data exploration and preparation for Chinese text classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the input data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible data splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level, such as INFO or DEBUG.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    if args.data_dir is not None:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.output_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.log_level is not None:
        overrides.setdefault("logging", {})["level"] = args.log_level

    if any(value is not None for value in (args.train_ratio, args.val_ratio, args.test_ratio)):
        split_overrides = overrides.setdefault("split", {})
        if args.train_ratio is not None:
            split_overrides["train_ratio"] = args.train_ratio
        if args.val_ratio is not None:
            split_overrides["val_ratio"] = args.val_ratio
        if args.test_ratio is not None:
            split_overrides["test_ratio"] = args.test_ratio

    return overrides


def load_dataset_rows(config: Dict[str, Any], logger) -> tuple[List[dict[str, Any]], List[str]]:
    data_dir = config["paths"]["data_dir"]
    encoding_candidates = config["processing"]["encoding_candidates"]
    rows: List[dict[str, Any]] = []
    failed_files: List[str] = []

    text_files = find_text_files(data_dir)
    logger.info("Discovered %d candidate text files under %s", len(text_files), data_dir)

    for file_path in text_files:
        label = extract_label_from_dirname(file_path.parent.name)
        if label not in LABEL2ID:
            logger.warning("Skipping file with unknown label folder: %s", file_path)
            continue

        text = read_text_file(file_path, encodings=encoding_candidates, logger=logger)
        if text is None:
            failed_files.append(_to_relative_path(file_path, config["project_root"]))
            continue

        cleaned_text = clean_text(text)
        if not cleaned_text:
            logger.warning("Text became empty after cleaning: %s", file_path)

        rows.append(
            {
                "text": cleaned_text,
                "label": label,
                "label_id": LABEL2ID[label],
                "file_path": _to_relative_path(file_path, config["project_root"]),
            }
        )

    rows.sort(key=lambda item: (item["label_id"], item["file_path"]))
    return rows, failed_files


def _to_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def validate_label_coverage(rows: Sequence[dict[str, Any]]) -> None:
    observed_labels = {row["label"] for row in rows}
    expected_labels = set(LABEL2ID)
    missing_labels = expected_labels - observed_labels
    unexpected_labels = observed_labels - expected_labels

    if missing_labels or unexpected_labels:
        raise ValueError(
            "Label coverage mismatch. "
            f"Missing labels: {sorted(missing_labels)}; "
            f"Unexpected labels: {sorted(unexpected_labels)}."
        )


def build_class_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    total = len(rows)
    class_counts = {label: 0 for label in LABEL2ID}
    for row in rows:
        class_counts[row["label"]] += 1

    return {
        label: {
            "label_id": LABEL2ID[label],
            "count": class_counts[label],
            "ratio": round(class_counts[label] / total, 6) if total else 0.0,
        }
        for label in LABEL2ID
    }


def compute_split_counts(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    raw_counts = [
        sample_count * train_ratio,
        sample_count * val_ratio,
        sample_count * test_ratio,
    ]
    counts = [math.floor(value) for value in raw_counts]
    remainder = sample_count - sum(counts)

    fractional_order = sorted(
        range(3),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1

    if sample_count >= 3:
        for split_index in range(3):
            if counts[split_index] == 0:
                donor_index = max(range(3), key=lambda index: counts[index])
                if counts[donor_index] <= 1:
                    raise ValueError(
                        "Unable to guarantee at least one sample per split for a label."
                    )
                counts[donor_index] -= 1
                counts[split_index] += 1

    train_count, val_count, test_count = counts
    if train_count + val_count + test_count != sample_count:
        raise AssertionError("Split counts do not sum to the original sample count.")
    return train_count, val_count, test_count


def stratified_split(
    rows: Sequence[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[List[dict[str, Any]], List[dict[str, Any]], List[dict[str, Any]]]:
    buckets: dict[str, List[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["label"]].append(dict(row))

    train_rows: List[dict[str, Any]] = []
    val_rows: List[dict[str, Any]] = []
    test_rows: List[dict[str, Any]] = []

    rng = random.Random(seed)
    for label in LABEL2ID:
        label_rows = buckets[label]
        rng.shuffle(label_rows)

        train_count, val_count, test_count = compute_split_counts(
            sample_count=len(label_rows),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_rows.extend(label_rows[:train_count])
        val_rows.extend(label_rows[train_count : train_count + val_count])
        test_rows.extend(label_rows[train_count + val_count : train_count + val_count + test_count])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def validate_split_label_coverage(split_name: str, rows: Sequence[dict[str, Any]]) -> None:
    labels = {row["label"] for row in rows}
    missing_labels = set(LABEL2ID) - labels
    if missing_labels:
        raise ValueError(
            f"{split_name} split is missing labels: {sorted(missing_labels)}"
        )


def build_split_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {label: 0 for label in LABEL2ID}
    for row in rows:
        counts[row["label"]] += 1
    return {label: {"count": counts[label]} for label in LABEL2ID}


def build_statistics_payload(
    rows: Sequence[dict[str, Any]],
    failed_files: Sequence[str],
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    char_lengths = [len(row["text"]) for row in rows]
    statistics = {
        "total_samples": len(rows),
        "num_classes": len(LABEL2ID),
        "class_distribution": build_class_distribution(rows),
        "length_statistics": {
            "character_count": summarize_numeric_series(char_lengths),
        },
        "label2id": dict(LABEL2ID),
        "id2label": {str(index): label for label, index in LABEL2ID.items()},
        "split_statistics": {
            "train": {
                "num_samples": len(train_rows),
                "class_distribution": build_split_statistics(train_rows),
            },
            "val": {
                "num_samples": len(val_rows),
                "class_distribution": build_split_statistics(val_rows),
            },
            "test": {
                "num_samples": len(test_rows),
                "class_distribution": build_split_statistics(test_rows),
            },
        },
        "failed_files_count": len(failed_files),
        "failed_files": list(failed_files),
    }
    return statistics


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=build_overrides(args))
    logger = setup_logging(config["logging"]["level"])

    logger.info("Loading configuration from %s", args.config)
    ensure_directories(
        [
            config["paths"]["output_dir"],
            config["paths"]["data_output_dir"],
            config["paths"]["figures_dir"],
            config["paths"]["results_dir"],
        ]
    )
    set_seed(config["seed"])

    logger.info("Reading and cleaning raw text files")
    rows, failed_files = load_dataset_rows(config, logger)
    validate_label_coverage(rows)
    logger.info("Successfully loaded %d samples", len(rows))
    if failed_files:
        logger.warning("Encountered %d unreadable files", len(failed_files))

    fieldnames = ["text", "label", "label_id", "file_path"]
    write_csv(rows, fieldnames, config["paths"]["all_data_csv"])
    logger.info("Saved all samples to %s", config["paths"]["all_data_csv"])

    logger.info("Creating stratified train/val/test split")
    train_rows, val_rows, test_rows = stratified_split(
        rows=rows,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=config["seed"],
    )
    validate_split_label_coverage("train", train_rows)
    validate_split_label_coverage("val", val_rows)
    validate_split_label_coverage("test", test_rows)

    # write_csv(train_rows, fieldnames, config["paths"]["train_csv"])
    # write_csv(val_rows, fieldnames, config["paths"]["val_csv"])
    # write_csv(test_rows, fieldnames, config["paths"]["test_csv"])
    logger.info(
        "Saved train/val/test splits to %s, %s, %s",
        config["paths"]["train_csv"],
        config["paths"]["val_csv"],
        config["paths"]["test_csv"],
    )

    statistics = build_statistics_payload(
        rows=rows,
        failed_files=failed_files,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )
    save_json(statistics, config["paths"]["statistics_json"])
    logger.info("Saved statistics to %s", config["paths"]["statistics_json"])

    ordered_counts = {
        label: statistics["class_distribution"][label]["count"]  # type: ignore[index]
        for label in LABEL2ID
    }
    char_lengths = [len(row["text"]) for row in rows]
    save_class_distribution_chart(
        ordered_counts,
        config["paths"]["class_distribution_png"],
        width=config["plot"]["width"],
        height=config["plot"]["height"],
    )
    save_length_distribution_chart(
        char_lengths,
        config["paths"]["length_distribution_png"],
        width=config["plot"]["width"],
        height=config["plot"]["height"],
        bins=config["plot"]["histogram_bins"],
    )
    logger.info(
        "Saved figures to %s and %s",
        config["paths"]["class_distribution_png"],
        config["paths"]["length_distribution_png"],
    )

    logger.info("Stage 1 data exploration completed successfully")


if __name__ == "__main__":
    main()
