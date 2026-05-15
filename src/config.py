from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "cfg" / "data_prepare.yaml"

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

DEFAULT_CONFIG: Dict[str, Any] = {
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


def _finalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
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

    config["project_root"] = PROJECT_ROOT
    config["label2id"] = copy.deepcopy(LABEL2ID)
    config["id2label"] = copy.deepcopy(ID2LABEL)
    return config


def load_config(
    config_path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        _deep_merge(merged, loaded)

    if overrides:
        _deep_merge(merged, overrides)

    return _finalize_config(merged)
