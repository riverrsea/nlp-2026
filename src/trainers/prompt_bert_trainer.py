from __future__ import annotations

from math import exp

from trainers.base_trainer import BaseTrainer


class PromptBertTrainer(BaseTrainer):
    trainer_name = "prompt_bert"

    def _require_runtime(self):
        try:
            import torch
            from transformers import AutoModelForMaskedLM
        except Exception as exc:
            raise ImportError(
                "torch and transformers are required for Prompt-BERT inference. "
                "Install them with `pip install torch transformers`."
            ) from exc
        return torch, AutoModelForMaskedLM

    def _load_tokenizer_and_model(self):
        from dataset import load_bert_tokenizer

        torch_module, auto_model_cls = self._require_runtime()
        del torch_module

        model_name_or_path = self.config.get("model_name_or_path") or "bert-base-chinese"
        try:
            tokenizer = load_bert_tokenizer(model_name_or_path=model_name_or_path)
            model = auto_model_cls.from_pretrained(model_name_or_path)
        except Exception as exc:
            raise ValueError(
                "Failed to load the HuggingFace tokenizer/model for Prompt-BERT from "
                f"`{model_name_or_path}`. If the current environment cannot download "
                "models from HuggingFace, please provide a local model path through "
                "`--model_name_or_path /path/to/local/model`."
            ) from exc
        return tokenizer, model

    def _build_examples_prefix(self, few_shot_examples):
        if not few_shot_examples:
            return ""

        verbalizer = self.model_config["verbalizer"]
        lines = ["下面是一些带标签的示例："]
        for example in few_shot_examples:
            label_word = verbalizer[str(example["label"])]
            lines.append(f"文本：{example['text']}")
            lines.append(f"类别：{label_word}")
        lines.append("")
        return "\n".join(lines)

    def _prepare_verbalizer_token_ids(self, tokenizer):
        verbalizer = self.model_config["verbalizer"]
        token_ids: dict[int, str] = {}
        for label, label_word in verbalizer.items():
            pieces = tokenizer.encode(label_word, add_special_tokens=False)
            if not pieces:
                raise ValueError(f"Failed to tokenize label word: {label_word}")
            token_ids[self.label2id[label]] = label_word
            self.logger.info(
                "Prompt-BERT verbalizer `%s` uses first token id %s for label `%s`",
                label_word,
                pieces[0],
                label,
            )
        return {
            self.label2id[label]: tokenizer.encode(label_word, add_special_tokens=False)[0]
            for label, label_word in verbalizer.items()
        }

    def _build_prompt(self, text: str, examples_prefix: str, mask_token: str) -> str:
        template = str(self.model_config["prompt_template"])
        prompt_body = template.replace("[MASK]", mask_token).format(text=text)
        return f"{examples_prefix}{prompt_body}" if examples_prefix else prompt_body

    def _predict_rows(self, rows, tokenizer, model, few_shot_examples):
        torch_module, _ = self._require_runtime()
        examples_prefix = self._build_examples_prefix(few_shot_examples)
        verbalizer_token_ids = self._prepare_verbalizer_token_ids(tokenizer)
        prompts = [
            self._build_prompt(str(row["text"]), examples_prefix, tokenizer.mask_token)
            for row in rows
        ]

        batch_size = int(self.train_config.get("batch_size", 8))
        predictions: list[int] = []
        scores: list[dict[str, float]] = []
        model.eval()

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=int(self.data_config.get("max_len", 256)),
                return_tensors="pt",
            )
            encoded = {key: value.to(self.config["device"]) for key, value in encoded.items()}
            input_ids = encoded["input_ids"]
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
            if mask_positions.size(0) != len(batch_prompts):
                raise ValueError(
                    "Each Prompt-BERT sample must contain exactly one [MASK] token after tokenization."
                )

            with torch_module.no_grad():
                logits = model(**encoded).logits

            for row_index in range(len(batch_prompts)):
                mask_position = int(mask_positions[row_index, 1].item())
                mask_logits = logits[row_index, mask_position]
                candidate_scores = {
                    label_id: float(mask_logits[token_id].item())
                    for label_id, token_id in verbalizer_token_ids.items()
                }
                max_logit = max(candidate_scores.values())
                normalized = {
                    label_id: exp(score - max_logit)
                    for label_id, score in candidate_scores.items()
                }
                denominator = sum(normalized.values()) or 1.0
                normalized = {
                    label_id: value / denominator
                    for label_id, value in normalized.items()
                }
                predicted_label_id = max(normalized, key=normalized.get)
                predictions.append(predicted_label_id)
                scores.append(
                    {
                        self.id2label[label_id]: float(probability)
                        for label_id, probability in normalized.items()
                    }
                )
        return predictions, scores

    def _run_prompt_evaluation(self, include_val: bool):
        datasets = self.load_train_val_test() if include_val else self.load_test_only()
        tokenizer, model = self._load_tokenizer_and_model()
        model.to(self.config["device"])

        train_rows = datasets.get("train", [])
        few_shot_k = int(self.prompt_config.get("few_shot_k", 0))
        few_shot_examples = self.sample_few_shot_examples(train_rows, few_shot_k) if train_rows else []

        metrics_by_split = {}
        predictions_by_split = {}
        scores_by_split = {}
        for split_name in ["val", "test"] if include_val else ["test"]:
            rows = datasets[split_name]
            predictions, scores = self._predict_rows(
                rows=rows,
                tokenizer=tokenizer,
                model=model,
                few_shot_examples=few_shot_examples,
            )
            _, labels = self.extract_texts_and_labels(rows)
            metrics_by_split[split_name] = self.compute_detailed_metrics(labels, predictions)
            predictions_by_split[split_name] = predictions
            scores_by_split[split_name] = scores
        return datasets, metrics_by_split, predictions_by_split, scores_by_split, few_shot_examples

    def train(self) -> dict[str, object]:
        datasets, metrics_by_split, predictions_by_split, scores_by_split, few_shot_examples = self._run_prompt_evaluation(include_val=True)
        prediction_path = self.save_predictions(datasets["test"], predictions_by_split["test"])
        result_payload = self.build_result_payload(
            mode="train",
            datasets=datasets,
            metrics_by_split=metrics_by_split,
            extra={
                "prediction_path": str(prediction_path),
                "few_shot_k": int(self.prompt_config.get("few_shot_k", 0)),
                "few_shot_examples_count": len(few_shot_examples),
                "score_preview": scores_by_split["test"][:5],
                "model_name_or_path": self.config.get("model_name_or_path"),
            },
        )
        results_path = self.save_results(result_payload)
        self.logger.info("Saved Prompt-BERT results to %s", results_path)
        self.logger.info("Saved Prompt-BERT predictions to %s", prediction_path)
        return {
            "status": "success",
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }

    def test(self) -> dict[str, object]:
        datasets, metrics_by_split, predictions_by_split, scores_by_split, _ = self._run_prompt_evaluation(include_val=False)
        prediction_path = self.save_predictions(datasets["test"], predictions_by_split["test"])
        result_payload = self.build_result_payload(
            mode="test",
            datasets=datasets,
            metrics_by_split=metrics_by_split,
            extra={
                "prediction_path": str(prediction_path),
                "few_shot_k": int(self.prompt_config.get("few_shot_k", 0)),
                "score_preview": scores_by_split["test"][:5],
                "model_name_or_path": self.config.get("model_name_or_path"),
            },
        )
        results_path = self.save_results(result_payload)
        self.logger.info("Saved Prompt-BERT test results to %s", results_path)
        self.logger.info("Saved Prompt-BERT test predictions to %s", prediction_path)
        return {
            "status": "success",
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }
