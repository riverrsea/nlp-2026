from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import (
    compute_classification_metrics,
    compute_detailed_classification_metrics,
    select_metrics,
)
from utils import (
    read_csv_rows,
    save_json,
    save_metric_curve,
    save_prediction_rows,
    save_training_curve,
    summarize_rows_by_label,
)


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
        self.prompt_config = self.config.get("prompt", {})
        self.experiment_name = self.config["experiment_name"]
        self.output_name = self.config.get("output_name") or self.experiment_name
        self.model_name = self.config["model_name"]
        self.label2id = self.config["label2id"]
        self.id2label = self.config["id2label"]

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

    def extract_texts_and_labels(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[list[str], list[int]]:
        texts = [str(row["text"]) for row in rows]
        labels = [int(row["label_id"]) for row in rows]
        return texts, labels

    def compute_summary_metrics(
        self,
        labels: Sequence[int],
        predictions: Sequence[int],
    ) -> dict[str, float]:
        metrics = compute_classification_metrics(
            y_true=labels,
            y_pred=predictions,
            labels=sorted(self.id2label.keys()),
        )
        return select_metrics(metrics, self.eval_config.get("metrics"))

    def compute_detailed_metrics(
        self,
        labels: Sequence[int],
        predictions: Sequence[int],
        extra_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, object]:
        metrics = compute_detailed_classification_metrics(
            y_true=labels,
            y_pred=predictions,
            labels=sorted(self.id2label.keys()),
            id2label=self.id2label,
        )
        selected_summary = select_metrics(metrics, self.eval_config.get("metrics"))
        payload = dict(metrics)
        payload.update({key: float(value) for key, value in selected_summary.items()})
        if extra_metrics:
            payload.update({key: float(value) for key, value in extra_metrics.items()})
        return payload

    def save_predictions(
        self,
        rows: Sequence[Mapping[str, object]],
        predictions: Sequence[int],
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path or self.paths["predictions_file"])
        save_prediction_rows(rows, predictions, self.id2label, path)
        return path

    def save_results(
        self,
        payload: Mapping[str, object],
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path or self.paths["results_file"])
        save_json(payload, path)
        if output_path is None and isinstance(payload, Mapping):
            mode = payload.get("mode")
            if mode == "train":
                save_json(payload, self.paths["train_results_file"])
            elif mode == "test":
                save_json(payload, self.paths["test_results_file"])
        return path

    def save_history_plot(
        self,
        history: Mapping[str, Sequence[float]],
        title: str,
    ) -> Path:
        path = Path(self.paths["training_curve_file"])
        save_training_curve(history=history, output_path=path, title=title)
        return path

    def save_single_metric_plot(
        self,
        values: Sequence[float],
        output_path: str | Path,
        title: str,
        y_label: str,
    ) -> Path:
        path = Path(output_path)
        save_metric_curve(values=values, output_path=path, title=title, y_label=y_label)
        return path

    def select_best_metric(
        self,
        metric_name: str,
        current_metrics: Mapping[str, float],
        best_score: float | None,
    ) -> tuple[bool, float]:
        if metric_name not in current_metrics:
            raise ValueError(
                f"Metric `{metric_name}` not found in computed metrics: {sorted(current_metrics)}"
            )
        current_score = float(current_metrics[metric_name])
        if best_score is None or current_score > best_score:
            return True, current_score
        return False, best_score

    def build_result_payload(
        self,
        mode: str,
        datasets: Mapping[str, Sequence[Mapping[str, object]]],
        metrics_by_split: Mapping[str, Mapping[str, float]],
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "success",
            "mode": mode,
            "experiment_name": self.experiment_name,
            "output_name": self.output_name,
            "model_name": self.model_name,
            "data_summary": self.summarize_datasets(datasets),
            "metrics": {split_name: dict(metrics) for split_name, metrics in metrics_by_split.items()},
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def sample_few_shot_examples(
        self,
        train_rows: Sequence[Mapping[str, object]],
        k: int,
    ) -> list[dict[str, object]]:
        if k <= 0:
            return []

        rng = random.Random(int(self.config["seed"]))
        buckets: dict[str, list[dict[str, object]]] = {label: [] for label in self.label2id}
        for row in train_rows:
            buckets[str(row["label"])].append(dict(row))

        sampled: list[dict[str, object]] = []
        for label in self.label2id:
            candidates = list(buckets[label])
            if not candidates:
                continue
            rng.shuffle(candidates)
            sampled.extend(candidates[: min(k, len(candidates))])
        return sampled

    def train(self) -> dict[str, object]:
        raise NotImplementedError

    def test(self) -> dict[str, object]:
        raise NotImplementedError
