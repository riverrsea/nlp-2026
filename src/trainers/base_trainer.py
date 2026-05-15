from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from utils import read_csv_rows, save_json, summarize_rows_by_label


class BaseTrainer:
    trainer_name = "base"

    def __init__(self, config: Mapping[str, Any], logger) -> None:
        self.config = dict(config)
        self.logger = logger
        self.paths = self.config["paths"]
        self.data_config = self.config["data"]
        self.model_config = self.config.get("model", {})
        self.train_config = self.config.get("train", {})
        self.eval_config = self.config.get("eval", {})
        self.experiment_name = self.config["experiment_name"]
        self.model_name = self.config["model_name"]

    def _load_split(self, split_name: str) -> list[dict[str, str]]:
        split_path = Path(self.data_config[f"{split_name}_file"])
        if not split_path.exists():
            raise FileNotFoundError(
                f"{split_name}.csv not found: {split_path}. "
                "Please run `python src/data_prepare.py` first to generate train.csv, val.csv and test.csv."
            )

        rows = read_csv_rows(split_path)
        if not rows:
            raise ValueError(f"{split_name}.csv is empty: {split_path}")

        required_columns = {"text", "label", "label_id", "file_path"}
        missing_columns = required_columns - set(rows[0].keys())
        if missing_columns:
            raise ValueError(
                f"{split_name}.csv is missing required columns: {sorted(missing_columns)}"
            )
        return rows

    def load_train_val_test(self) -> dict[str, list[dict[str, str]]]:
        return {
            "train": self._load_split("train"),
            "val": self._load_split("val"),
            "test": self._load_split("test"),
        }

    def load_test_only(self) -> dict[str, list[dict[str, str]]]:
        return {"test": self._load_split("test")}

    def require_checkpoint(self) -> Path:
        checkpoint = self.config.get("checkpoint")
        if checkpoint is None:
            raise ValueError(
                "`--checkpoint` is required in mode=test. "
                "Please provide a checkpoint path with `--checkpoint path/to/model.ckpt`."
            )

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please check the checkpoint path passed to `--checkpoint`."
            )
        return checkpoint_path

    def summarize_datasets(
        self,
        datasets: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> dict[str, dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for split_name, rows in datasets.items():
            summary[split_name] = {
                "num_samples": len(rows),
                "class_distribution": summarize_rows_by_label(rows),
            }
        return summary

    def save_framework_note(
        self,
        mode: str,
        datasets: Mapping[str, Sequence[Mapping[str, object]]],
        message: str,
        extra_payload: Mapping[str, object] | None = None,
    ) -> Path:
        payload: dict[str, object] = {
            "status": "framework_ready",
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "mode": mode,
            "message": message,
            "data_summary": self.summarize_datasets(datasets),
        }
        if extra_payload:
            payload.update(dict(extra_payload))

        note_path = (
            Path(self.paths["train_framework_note_file"])
            if mode == "train"
            else Path(self.paths["test_framework_note_file"])
        )
        save_json(payload, note_path)
        return note_path

    def train(self) -> dict[str, object]:
        raise NotImplementedError

    def test(self) -> dict[str, object]:
        raise NotImplementedError
