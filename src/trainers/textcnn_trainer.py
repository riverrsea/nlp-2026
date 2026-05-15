from __future__ import annotations

from pathlib import Path

from dataset import TextCNNDataset, build_vocab, load_pretrained_word_vectors, tokenize_with_jieba
from models.textcnn import TextCNN
from trainers.base_trainer import BaseTrainer


class TextCNNTrainer(BaseTrainer):
    trainer_name = "textcnn"

    def _require_torch(self):
        try:
            import torch
            from torch.utils.data import DataLoader
        except Exception as exc:
            raise ImportError(
                "torch is required for the TextCNN model. "
                "Install it with `pip install torch`."
            ) from exc
        return torch, DataLoader

    def _validate_pretrained_path_if_needed(self, require_for_train: bool = True) -> Path | None:
        embedding_type = str(self.config.get("embedding_type", "random")).lower()
        if embedding_type != "pretrained":
            return None

        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path is None:
            if require_for_train:
                raise ValueError(
                    "TextCNN with pretrained embeddings requires `--pretrained_path` "
                    "or `pretrained_path` in the YAML config."
                )
            return None

        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained word vector file not found: {path}. "
                "Please provide a valid path through `--pretrained_path`."
            )
        return path

    def _build_model(self, vocabulary: dict[str, int], embedding_weights=None):
        model_kwargs = {
            "vocab_size": len(vocabulary),
            "embedding_dim": int(self.model_config.get("embedding_dim", 300)),
            "num_classes": int(self.model_config.get("num_classes", len(self.label2id))),
            "num_filters": int(self.model_config.get("num_filters", 128)),
            "kernel_sizes": tuple(self.model_config.get("kernel_sizes", [3, 4, 5])),
            "dropout": float(self.model_config.get("dropout", 0.5)),
            "padding_idx": vocabulary.get("<PAD>", 0),
            "embedding_weights": embedding_weights,
            "freeze_embeddings": bool(self.model_config.get("freeze_embeddings", False)),
        }
        return TextCNN(**model_kwargs), model_kwargs

    def _build_optimizer(self, torch_module, model):
        optimizer_name = str(self.train_config.get("optimizer", "adam")).lower()
        learning_rate = float(self.train_config.get("learning_rate", 0.001))
        if optimizer_name == "adam":
            return torch_module.optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_name == "adamw":
            return torch_module.optim.AdamW(model.parameters(), lr=learning_rate)
        if optimizer_name == "sgd":
            return torch_module.optim.SGD(model.parameters(), lr=learning_rate)
        raise ValueError(f"Unsupported optimizer for TextCNN: {optimizer_name}")

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool):
        torch_module, data_loader_cls = self._require_torch()
        del torch_module
        return data_loader_cls(dataset, batch_size=batch_size, shuffle=shuffle)

    def _run_epoch(self, torch_module, model, data_loader, optimizer, criterion, device: str) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        return total_loss / max(total_samples, 1)

    def _evaluate(self, torch_module, model, data_loader, criterion, device: str):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions: list[int] = []
        all_labels: list[int] = []

        with torch_module.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                predictions = torch_module.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        average_loss = total_loss / max(total_samples, 1)
        metrics = self.compute_summary_metrics(all_labels, all_predictions)
        metrics["loss"] = average_loss
        return metrics, all_predictions, all_labels

    def _save_checkpoint(
        self,
        torch_module,
        model,
        vocabulary: dict[str, int],
        model_kwargs: dict[str, object],
        best_epoch: int,
        best_metric: float,
    ) -> Path:
        checkpoint_path = Path(self.paths["best_checkpoint_file"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        clean_model_kwargs = dict(model_kwargs)
        clean_model_kwargs["embedding_weights"] = None
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "vocab": vocabulary,
            "model_kwargs": clean_model_kwargs,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "embedding_type": self.config.get("embedding_type", "random"),
            "experiment_name": self.experiment_name,
        }
        torch_module.save(checkpoint_payload, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint_model(self, checkpoint_path: Path):
        torch_module, _ = self._require_torch()
        checkpoint = torch_module.load(checkpoint_path, map_location=self.config["device"])
        model_kwargs = dict(checkpoint["model_kwargs"])
        model_kwargs["embedding_weights"] = None
        model = TextCNN(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.config["device"])
        return model, checkpoint

    def train(self) -> dict[str, object]:
        torch_module, _ = self._require_torch()
        pretrained_path = self._validate_pretrained_path_if_needed(require_for_train=True)
        datasets = self.load_train_val_test()
        train_texts, train_labels = self.extract_texts_and_labels(datasets["train"])
        val_texts, val_labels = self.extract_texts_and_labels(datasets["val"])
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])

        max_vocab_size = self.model_config.get("max_vocab_size")
        vocabulary = build_vocab(
            texts=train_texts,
            tokenizer=tokenize_with_jieba,
            min_freq=int(self.model_config.get("vocab_min_freq", 1)),
            max_vocab_size=int(max_vocab_size) if max_vocab_size is not None else None,
        )

        embedding_weights = None
        if pretrained_path is not None:
            embedding_weights = load_pretrained_word_vectors(
                embedding_path=pretrained_path,
                vocabulary=vocabulary,
                embedding_dim=int(self.model_config.get("embedding_dim", 300)),
            )

        max_len = int(self.data_config.get("max_len", 512))
        train_dataset = TextCNNDataset(
            texts=train_texts,
            labels=train_labels,
            vocabulary=vocabulary,
            min_freq=int(self.model_config.get("vocab_min_freq", 1)),
            max_vocab_size=int(max_vocab_size) if max_vocab_size is not None else None,
            max_length=max_len,
        )
        val_dataset = TextCNNDataset(
            texts=val_texts,
            labels=val_labels,
            vocabulary=vocabulary,
            max_length=max_len,
        )
        test_dataset = TextCNNDataset(
            texts=test_texts,
            labels=test_labels,
            vocabulary=vocabulary,
            max_length=max_len,
        )

        batch_size = int(self.train_config.get("batch_size", 32))
        train_loader = self._create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        model, model_kwargs = self._build_model(vocabulary, embedding_weights=embedding_weights)
        model.to(self.config["device"])
        optimizer = self._build_optimizer(torch_module, model)
        criterion = torch_module.nn.CrossEntropyLoss()

        metric_name = str(self.train_config.get("metric_for_best_model", "macro_f1"))
        epochs = int(self.train_config.get("epochs", 3))
        patience = int(self.train_config.get("early_stop_patience", 3))
        best_score = None
        best_epoch = 0
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

        self.logger.info(
            "Training TextCNN (%s) with vocab_size=%d, max_len=%d, epochs=%d",
            self.config.get("embedding_type", "random"),
            len(vocabulary),
            max_len,
            epochs,
        )

        checkpoint_path = Path(self.paths["best_checkpoint_file"])
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                torch_module=torch_module,
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.config["device"],
            )
            val_metrics, _, _ = self._evaluate(
                torch_module=torch_module,
                model=model,
                data_loader=val_loader,
                criterion=criterion,
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
                    vocabulary=vocabulary,
                    model_kwargs=model_kwargs,
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
            criterion=criterion,
            device=self.config["device"],
        )
        test_summary_metrics, test_predictions, test_eval_labels = self._evaluate(
            torch_module=torch_module,
            model=best_model,
            data_loader=test_loader,
            criterion=criterion,
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
                "embedding_type": self.config.get("embedding_type", "random"),
                "vocab_size": len(vocabulary),
                "history": history,
            },
        )
        results_path = self.save_results(results_payload)
        history_path = self.save_results(history, self.paths["history_file"])
        curve_path = self.save_history_plot(
            history=history,
            title=f"TextCNN ({self.config.get('embedding_type', 'random')})",
        )
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

        self.logger.info("Saved TextCNN checkpoint to %s", checkpoint_path)
        self.logger.info("Saved TextCNN results to %s", results_path)
        self.logger.info("Saved TextCNN predictions to %s", prediction_path)
        self.logger.info("Saved TextCNN history to %s", history_path)
        self.logger.info("Saved TextCNN curve to %s", curve_path)
        self.logger.info("Saved TextCNN loss curve to %s", loss_curve_path)
        self.logger.info("Saved TextCNN accuracy curve to %s", accuracy_curve_path)
        return {
            "status": "success",
            "checkpoint_path": str(checkpoint_path),
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }

    def test(self) -> dict[str, object]:
        torch_module, _ = self._require_torch()
        checkpoint_path = self.require_checkpoint()
        model, checkpoint = self._load_checkpoint_model(checkpoint_path)
        datasets = self.load_test_only()
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])
        vocabulary = checkpoint["vocab"]
        max_len = int(self.data_config.get("max_len", 512))
        test_dataset = TextCNNDataset(
            texts=test_texts,
            labels=test_labels,
            vocabulary=vocabulary,
            max_length=max_len,
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=int(self.train_config.get("batch_size", 32)),
            shuffle=False,
        )
        criterion = torch_module.nn.CrossEntropyLoss()
        test_summary_metrics, test_predictions, test_eval_labels = self._evaluate(
            torch_module=torch_module,
            model=model,
            data_loader=test_loader,
            criterion=criterion,
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
                "embedding_type": checkpoint.get("embedding_type", self.config.get("embedding_type", "random")),
            },
        )
        results_path = self.save_results(results_payload)
        self.logger.info("Saved TextCNN test results to %s", results_path)
        self.logger.info("Saved TextCNN test predictions to %s", prediction_path)
        return {
            "status": "success",
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }
