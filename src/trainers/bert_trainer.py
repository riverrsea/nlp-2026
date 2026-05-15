from __future__ import annotations

from pathlib import Path

from dataset import BertTextDataset, load_bert_tokenizer
from trainers.base_trainer import BaseTrainer


class BertTrainer(BaseTrainer):
    trainer_name = "bert"

    def _require_runtime(self):
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification
        except Exception as exc:
            raise ImportError(
                "torch and transformers are required for BERT fine-tuning. "
                "Install them with `pip install torch transformers`."
            ) from exc
        return torch, DataLoader, AutoModelForSequenceClassification

    def _load_tokenizer_and_model(self):
        torch_module, _, auto_model_cls = self._require_runtime()
        del torch_module

        model_name_or_path = self.config.get("model_name_or_path") or "bert-base-chinese"
        try:
            tokenizer = load_bert_tokenizer(model_name_or_path=model_name_or_path)
            model = auto_model_cls.from_pretrained(
                model_name_or_path,
                num_labels=int(self.model_config.get("num_classes", len(self.label2id))),
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load the HuggingFace tokenizer/model from "
                f"`{model_name_or_path}`. If the current environment cannot download "
                "models from HuggingFace, please provide a local model path through "
                "`--model_name_or_path /path/to/local/model`."
            ) from exc
        return tokenizer, model

    def _build_optimizer(self, torch_module, model):
        learning_rate = float(self.train_config.get("learning_rate", 2e-5))
        return torch_module.optim.AdamW(model.parameters(), lr=learning_rate)

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool):
        _, data_loader_cls, _ = self._require_runtime()
        return data_loader_cls(dataset, batch_size=batch_size, shuffle=shuffle)

    def _run_epoch(self, torch_module, model, data_loader, optimizer, device: str) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in data_loader:
            labels = batch["label"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "label"
            }
            optimizer.zero_grad()
            outputs = model(**model_inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        return total_loss / max(total_samples, 1)

    def _evaluate(self, torch_module, model, data_loader, device: str):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions: list[int] = []
        all_labels: list[int] = []
        with torch_module.no_grad():
            for batch in data_loader:
                labels = batch["label"].to(device)
                model_inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != "label"
                }
                outputs = model(**model_inputs, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                predictions = torch_module.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = self.compute_summary_metrics(all_labels, all_predictions)
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics, all_predictions, all_labels

    def _save_checkpoint(self, torch_module, model, tokenizer_name: str, best_epoch: int, best_metric: float) -> Path:
        checkpoint_path = Path(self.paths["best_checkpoint_file"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name_or_path": tokenizer_name,
                "num_labels": int(self.model_config.get("num_classes", len(self.label2id))),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _load_checkpoint_model(self, checkpoint_path: Path):
        torch_module, _, auto_model_cls = self._require_runtime()
        checkpoint = torch_module.load(checkpoint_path, map_location=self.config["device"])
        model_name_or_path = checkpoint["model_name_or_path"]
        try:
            model = auto_model_cls.from_pretrained(
                model_name_or_path,
                num_labels=int(checkpoint["num_labels"]),
            )
        except Exception as exc:
            raise ValueError(
                "Failed to rebuild the BERT model from checkpoint metadata. "
                f"Please make sure `{model_name_or_path}` is available locally or can be downloaded."
            ) from exc
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.config["device"])
        return model, checkpoint

    def train(self) -> dict[str, object]:
        torch_module, _, _ = self._require_runtime()
        datasets = self.load_train_val_test()
        train_texts, train_labels = self.extract_texts_and_labels(datasets["train"])
        val_texts, val_labels = self.extract_texts_and_labels(datasets["val"])
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])

        tokenizer, model = self._load_tokenizer_and_model()
        model.to(self.config["device"])

        max_len = int(self.data_config.get("max_len", 256))
        train_dataset = BertTextDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=max_len,
        )
        val_dataset = BertTextDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=tokenizer,
            max_length=max_len,
        )
        test_dataset = BertTextDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer,
            max_length=max_len,
        )

        batch_size = int(self.train_config.get("batch_size", 16))
        train_loader = self._create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = self._build_optimizer(torch_module, model)
        metric_name = str(self.train_config.get("metric_for_best_model", "macro_f1"))
        epochs = int(self.train_config.get("epochs", 1))
        patience = int(self.train_config.get("early_stop_patience", 2))
        best_score = None
        best_epoch = 0
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }
        checkpoint_path = Path(self.paths["best_checkpoint_file"])

        self.logger.info(
            "Training BERT model `%s` with max_len=%d, epochs=%d",
            self.config.get("model_name_or_path"),
            max_len,
            epochs,
        )
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                torch_module=torch_module,
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=self.config["device"],
            )
            val_metrics, _, _ = self._evaluate(
                torch_module=torch_module,
                model=model,
                data_loader=val_loader,
                device=self.config["device"],
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_accuracy"].append(float(val_metrics.get("accuracy", 0.0)))
            history["val_macro_f1"].append(float(val_metrics.get("macro_f1", 0.0)))

            improved, best_score = self.select_best_metric(metric_name, val_metrics, best_score)
            self.logger.info(
                "Epoch %d/%d - train_loss=%.4f val_%s=%.4f",
                epoch,
                epochs,
                train_loss,
                metric_name,
                float(val_metrics[metric_name]),
            )
            if improved:
                best_epoch = epoch
                patience_counter = 0
                checkpoint_path = self._save_checkpoint(
                    torch_module=torch_module,
                    model=model,
                    tokenizer_name=str(self.config.get("model_name_or_path")),
                    best_epoch=best_epoch,
                    best_metric=float(best_score),
                )
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    self.logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        best_model, checkpoint = self._load_checkpoint_model(checkpoint_path)
        val_summary_metrics, val_predictions, val_eval_labels = self._evaluate(
            torch_module=torch_module,
            model=best_model,
            data_loader=val_loader,
            device=self.config["device"],
        )
        test_summary_metrics, test_predictions, test_eval_labels = self._evaluate(
            torch_module=torch_module,
            model=best_model,
            data_loader=test_loader,
            device=self.config["device"],
        )
        val_metrics = self.compute_detailed_metrics(
            val_eval_labels,
            val_predictions,
            extra_metrics={"loss": float(val_summary_metrics["loss"])},
        )
        test_metrics = self.compute_detailed_metrics(
            test_eval_labels,
            test_predictions,
            extra_metrics={"loss": float(test_summary_metrics["loss"])},
        )

        prediction_path = self.save_predictions(datasets["test"], test_predictions)
        results_payload = self.build_result_payload(
            mode="train",
            datasets=datasets,
            metrics_by_split={"val": val_metrics, "test": test_metrics},
            extra={
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_metric": float(checkpoint["best_metric"]),
                "history": history,
                "model_name_or_path": self.config.get("model_name_or_path"),
            },
        )
        results_path = self.save_results(results_payload)
        history_path = self.save_results(history, self.paths["history_file"])
        curve_path = self.save_history_plot(history=history, title=f"BERT {self.experiment_name}")
        loss_curve_path = self.save_single_metric_plot(
            values=history["val_loss"],
            output_path=Path(self.paths["figures_dir"]) / f"{self.output_name}_loss.png",
            title=f"{self.output_name} Validation Loss",
            y_label="Loss",
        )
        accuracy_curve_path = self.save_single_metric_plot(
            values=history["val_accuracy"],
            output_path=Path(self.paths["figures_dir"]) / f"{self.output_name}_accuracy.png",
            title=f"{self.output_name} Validation Accuracy",
            y_label="Accuracy",
        )

        self.logger.info("Saved BERT checkpoint to %s", checkpoint_path)
        self.logger.info("Saved BERT results to %s", results_path)
        self.logger.info("Saved BERT predictions to %s", prediction_path)
        self.logger.info("Saved BERT history to %s", history_path)
        self.logger.info("Saved BERT curve to %s", curve_path)
        self.logger.info("Saved BERT loss curve to %s", loss_curve_path)
        self.logger.info("Saved BERT accuracy curve to %s", accuracy_curve_path)
        return {
            "status": "success",
            "checkpoint_path": str(checkpoint_path),
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }

    def test(self) -> dict[str, object]:
        torch_module, _, _ = self._require_runtime()
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])

        model, checkpoint = self._load_checkpoint_model(checkpoint_path)
        try:
            tokenizer = load_bert_tokenizer(
                model_name_or_path=str(checkpoint["model_name_or_path"])
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load the tokenizer required by the saved BERT checkpoint. "
                f"Please make sure `{checkpoint['model_name_or_path']}` is available locally."
            ) from exc
        test_dataset = BertTextDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer,
            max_length=int(self.data_config.get("max_len", 256)),
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=int(self.train_config.get("batch_size", 16)),
            shuffle=False,
        )
        test_summary_metrics, test_predictions, test_eval_labels = self._evaluate(
            torch_module=torch_module,
            model=model,
            data_loader=test_loader,
            device=self.config["device"],
        )
        test_metrics = self.compute_detailed_metrics(
            test_eval_labels,
            test_predictions,
            extra_metrics={"loss": float(test_summary_metrics["loss"])},
        )
        prediction_path = self.save_predictions(datasets["test"], test_predictions)
        results_payload = self.build_result_payload(
            mode="test",
            datasets=datasets,
            metrics_by_split={"test": test_metrics},
            extra={
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_metric": float(checkpoint["best_metric"]),
            },
        )
        results_path = self.save_results(results_payload)
        self.logger.info("Saved BERT test results to %s", results_path)
        self.logger.info("Saved BERT test predictions to %s", prediction_path)
        return {
            "status": "success",
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }
