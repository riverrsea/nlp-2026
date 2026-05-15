from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "cfg" / "data_prepare.yaml"
DEFAULT_DATA_PREPARE_CONFIG_PATH = DEFAULT_CONFIG_PATH

LABEL2ID = {
    "art": 0,
    "computer": 1,
    "economy": 2,
    "transportation": 3,
    "education": 4,
    "environment": 5,
    "sports": 6,
    "military": 7,
    "politics": 8,
    "medicine": 9,
}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

DEFAULT_DATA_PREPARE_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "data_dir": "data",
        "output_dir": "outputs",
        "data_subdir": "data",
        "figures_subdir": "figures",
        "results_subdir": "results",
        "all_data_filename": "all_data.csv",
        "train_filename": "train.csv",
        "val_filename": "val.csv",
        "test_filename": "test.csv",
        "statistics_filename": "data_statistics.json",
        "class_distribution_figure": "class_distribution.png",
        "length_distribution_figure": "length_distribution.png",
    },
    "split": {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "processing": {
        "encoding_candidates": ["utf-8", "gb18030", "gbk"],
        "textcnn": {
            "min_freq": 1,
            "max_vocab_size": 30000,
            "max_length": 256,
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
        },
        "bert": {
            "model_name_or_path": "bert-base-chinese",
            "max_length": 256,
            "padding": "max_length",
            "truncation": True,
        },
    },
    "plot": {
        "width": 1400,
        "height": 900,
        "histogram_bins": 30,
    },
    "logging": {
        "level": "INFO",
    },
}

DEFAULT_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "experiment_name": "experiment",
    "model_name": "",
    "mode": "train",
    "seed": 42,
    "device": "auto",
    "checkpoint": None,
    "pretrained_path": None,
    "model_name_or_path": None,
    "paths": {
        "data_dir": "data",
        "output_dir": "outputs",
        "data_subdir": "data",
        "figures_subdir": "figures",
        "results_subdir": "results",
        "predictions_subdir": "predictions",
        "checkpoints_subdir": "checkpoints",
    },
    "data": {
        "train_file": "outputs/data/train.csv",
        "val_file": "outputs/data/val.csv",
        "test_file": "outputs/data/test.csv",
        "max_len": 512,
    },
    "model": {},
    "train": {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "early_stop_patience": 3,
        "metric_for_best_model": "macro_f1",
    },
    "eval": {
        "metrics": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ],
    },
    "logging": {
        "level": "INFO",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_if_local_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(_resolve_path(value))

    if value.startswith("/") or value.startswith("."):
        return str(_resolve_path(value))

    candidate = PROJECT_ROOT / value
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _validate_split(split_cfg: Dict[str, float]) -> None:
    total = (
        float(split_cfg["train_ratio"])
        + float(split_cfg["val_ratio"])
        + float(split_cfg["test_ratio"])
    )
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Split ratios must sum to 1.0, "
            f"but got train+val+test={total:.6f}."
        )


def _attach_label_maps(config: Dict[str, Any]) -> Dict[str, Any]:
    config["project_root"] = PROJECT_ROOT
    config["label2id"] = copy.deepcopy(LABEL2ID)
    config["id2label"] = copy.deepcopy(ID2LABEL)
    return config


def _finalize_data_prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config["paths"]
    _validate_split(config["split"])

    data_dir = _resolve_path(paths["data_dir"])
    output_dir = _resolve_path(paths["output_dir"])
    data_output_dir = output_dir / paths["data_subdir"]
    figures_dir = output_dir / paths["figures_subdir"]
    results_dir = output_dir / paths["results_subdir"]

    paths["data_dir"] = data_dir
    paths["output_dir"] = output_dir
    paths["data_output_dir"] = data_output_dir
    paths["figures_dir"] = figures_dir
    paths["results_dir"] = results_dir
    paths["all_data_csv"] = data_output_dir / paths["all_data_filename"]
    paths["train_csv"] = data_output_dir / paths["train_filename"]
    paths["val_csv"] = data_output_dir / paths["val_filename"]
    paths["test_csv"] = data_output_dir / paths["test_filename"]
    paths["statistics_json"] = results_dir / paths["statistics_filename"]
    paths["class_distribution_png"] = (
        figures_dir / paths["class_distribution_figure"]
    )
    paths["length_distribution_png"] = figures_dir / paths["length_distribution_figure"]
    return _attach_label_maps(config)


def _finalize_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not config.get("model_name"):
        raise ValueError("`model_name` must be defined in the YAML config.")

    experiment_name = config.get("experiment_name") or "experiment"
    paths = config.setdefault("paths", {})

    data_dir = _resolve_path(paths.get("data_dir", "data"))
    output_dir = _resolve_path(paths.get("output_dir", "outputs"))
    data_subdir = paths.get("data_subdir", "data")
    figures_subdir = paths.get("figures_subdir", "figures")
    results_subdir = paths.get("results_subdir", "results")
    predictions_subdir = paths.get("predictions_subdir", "predictions")
    checkpoints_subdir = paths.get("checkpoints_subdir", "checkpoints")

    data_output_dir = output_dir / data_subdir
    figures_dir = output_dir / figures_subdir
    results_dir = output_dir / results_subdir
    predictions_dir = output_dir / predictions_subdir
    checkpoints_dir = output_dir / checkpoints_subdir
    experiment_checkpoint_dir = checkpoints_dir / experiment_name

    paths["data_dir"] = data_dir
    paths["output_dir"] = output_dir
    paths["data_output_dir"] = data_output_dir
    paths["figures_dir"] = figures_dir
    paths["results_dir"] = results_dir
    paths["predictions_dir"] = predictions_dir
    paths["checkpoints_dir"] = checkpoints_dir
    paths["experiment_checkpoint_dir"] = experiment_checkpoint_dir
    paths["train_results_file"] = results_dir / f"{experiment_name}_train_results.json"
    paths["test_results_file"] = results_dir / f"{experiment_name}_test_results.json"
    paths["train_framework_note_file"] = (
        results_dir / f"{experiment_name}_train_framework.json"
    )
    paths["test_framework_note_file"] = (
        results_dir / f"{experiment_name}_test_framework.json"
    )
    paths["train_predictions_file"] = (
        predictions_dir / f"{experiment_name}_train_predictions.csv"
    )
    paths["test_predictions_file"] = (
        predictions_dir / f"{experiment_name}_test_predictions.csv"
    )
    paths["best_checkpoint_file"] = (
        experiment_checkpoint_dir / f"{experiment_name}_best.ckpt"
    )
    paths["history_file"] = results_dir / f"{experiment_name}_history.json"
    paths["training_curve_file"] = figures_dir / f"{experiment_name}_training_curve.png"

    data_cfg = config.setdefault("data", {})
    data_cfg["train_file"] = _resolve_path(data_cfg.get("train_file", "outputs/data/train.csv"))
    data_cfg["val_file"] = _resolve_path(data_cfg.get("val_file", "outputs/data/val.csv"))
    data_cfg["test_file"] = _resolve_path(data_cfg.get("test_file", "outputs/data/test.csv"))

    checkpoint = config.get("checkpoint")
    config["checkpoint"] = _resolve_path(checkpoint) if checkpoint else None

    pretrained_path = config.get("pretrained_path")
    config["pretrained_path"] = _resolve_path(pretrained_path) if pretrained_path else None

    model_name_or_path = config.get("model_name_or_path")
    if model_name_or_path is None:
        model_name_or_path = config.get("model", {}).get("model_name_or_path")
    config["model_name_or_path"] = _resolve_if_local_path(model_name_or_path)

    return _attach_label_maps(config)


def _load_yaml_file(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def load_data_prepare_config(
    config_path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_DATA_PREPARE_CONFIG)
    path = Path(config_path) if config_path is not None else DEFAULT_DATA_PREPARE_CONFIG_PATH

    if path.exists():
        loaded = _load_yaml_file(path)
        _deep_merge(merged, loaded)

    if overrides:
        _deep_merge(merged, overrides)

    return _finalize_data_prepare_config(merged)


def load_experiment_config(
    config_path: str | Path,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    loaded = _load_yaml_file(config_path)
    _deep_merge(merged, loaded)

    if overrides:
        _deep_merge(merged, overrides)

    return _finalize_experiment_config(merged)


def load_config(
    config_path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return load_data_prepare_config(config_path=config_path, overrides=overrides)
