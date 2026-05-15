from __future__ import annotations

import os

from trainers.base_trainer import BaseTrainer
from utils import write_csv


class PromptGPTTrainer(BaseTrainer):
    trainer_name = "prompt_gpt"

    def _build_examples_prefix(self, few_shot_examples):
        if not few_shot_examples:
            return ""

        label_names = {
            "art": "艺术",
            "computer": "计算机",
            "economy": "经济",
            "transportation": "交通",
            "education": "教育",
            "environment": "环境",
            "sports": "体育",
            "military": "军事",
            "politics": "政治",
            "medicine": "医药",
        }
        lines = ["下面是一些示例："]
        for example in few_shot_examples:
            lines.append(f"文本：{example['text']}")
            lines.append(f"答案：{label_names[str(example['label'])]}")
        lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, text: str, few_shot_examples) -> str:
        prompt_template = str(self.model_config["prompt_template"])
        prefix = self._build_examples_prefix(few_shot_examples)
        prompt = prompt_template.format(text=text)
        return f"{prefix}{prompt}" if prefix else prompt

    def _save_placeholder_predictions(self, rows) -> str:
        output_path = self.paths["predictions_file"]
        placeholder_rows = []
        for row in rows:
            placeholder_rows.append(
                {
                    "text": row.get("text", ""),
                    "label": row.get("label", ""),
                    "label_id": row.get("label_id", ""),
                    "pred_label": "",
                    "pred_label_id": "",
                    "file_path": row.get("file_path", ""),
                }
            )
        write_csv(
            placeholder_rows,
            fieldnames=[
                "text",
                "label",
                "label_id",
                "pred_label",
                "pred_label_id",
                "file_path",
            ],
            output_path=output_path,
        )
        return str(output_path)

    def _build_environment_message(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return (
                "Detected `OPENAI_API_KEY`, but the concrete GPT API invocation path has "
                "not been implemented yet. Extend `src/trainers/prompt_gpt_trainer.py` to "
                "call your preferred API or local generation service."
            )
        return (
            "GPT Prompt interface is ready, but no GPT runtime is configured. "
            "Please provide a local generation model or set `OPENAI_API_KEY`, then "
            "extend `src/trainers/prompt_gpt_trainer.py` to perform real inference."
        )

    def _run_interface(self, include_val: bool):
        datasets = self.load_train_val_test() if include_val else self.load_test_only()
        train_rows = datasets.get("train", [])
        few_shot_k = int(self.prompt_config.get("few_shot_k", 0))
        few_shot_examples = self.sample_few_shot_examples(train_rows, few_shot_k) if train_rows else []
        prompt_preview = self._build_prompt(str(datasets["test"][0]["text"]), few_shot_examples) if datasets["test"] else ""
        prediction_path = self._save_placeholder_predictions(datasets["test"])
        result_payload = {
            "status": "environment_required",
            "mode": "train" if include_val else "test",
            "experiment_name": self.experiment_name,
            "output_name": self.output_name,
            "model_name": self.model_name,
            "message": self._build_environment_message(),
            "data_summary": self.summarize_datasets(datasets),
            "few_shot_k": few_shot_k,
            "few_shot_examples_count": len(few_shot_examples),
            "prediction_path": prediction_path,
            "prompt_preview": prompt_preview[:2000],
            "integration_notes": [
                "Implement a local generation model or remote API client in `src/trainers/prompt_gpt_trainer.py`.",
                "Parse the generated Chinese label name into one of the 10 fixed categories.",
                "Then save real metrics and predictions through the shared BaseTrainer helpers.",
            ],
        }
        results_path = self.save_results(result_payload)
        self.logger.warning(result_payload["message"])
        self.logger.info("Saved GPT prompt interface note to %s", results_path)
        self.logger.info("Saved GPT placeholder predictions to %s", prediction_path)
        return {
            "status": "environment_required",
            "results_path": str(results_path),
            "prediction_path": prediction_path,
        }

    def train(self) -> dict[str, object]:
        return self._run_interface(include_val=True)

    def test(self) -> dict[str, object]:
        return self._run_interface(include_val=False)
