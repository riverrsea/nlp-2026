from __future__ import annotations

from pathlib import Path

from dataset import tokenize_with_jieba
from trainers.base_trainer import BaseTrainer


class SVMTrainer(BaseTrainer):
    trainer_name = "svm"

    def _build_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:
            raise ImportError(
                "scikit-learn is required for the TF-IDF + SVM baseline. "
                "Install it with `pip install scikit-learn`."
            ) from exc

        ngram_range = tuple(self.model_config.get("ngram_range", [1, 1]))
        return TfidfVectorizer(
            tokenizer=tokenize_with_jieba,
            token_pattern=None,
            lowercase=False,
            max_features=self.model_config.get("max_features"),
            ngram_range=ngram_range,
            min_df=self.model_config.get("min_df", 1),
        )

    def _build_classifier(self):
        try:
            from sklearn.svm import LinearSVC, SVC
        except Exception as exc:
            raise ImportError(
                "scikit-learn is required for the TF-IDF + SVM baseline. "
                "Install it with `pip install scikit-learn`."
            ) from exc

        c_value = float(self.model_config.get("c", 1.0))
        kernel = str(self.model_config.get("kernel", "linear")).lower()
        if kernel == "linear":
            return LinearSVC(C=c_value)
        return SVC(C=c_value, kernel=kernel)

    def _save_checkpoint(self, vectorizer, classifier) -> Path:
        checkpoint_path = Path(self.paths["best_checkpoint_file"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
        except Exception as exc:
            raise ImportError(
                "joblib is required for saving the SVM checkpoint. "
                "Install it with `pip install joblib`."
            ) from exc

        joblib.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "config": self.config,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path):
        try:
            import joblib
        except Exception as exc:
            raise ImportError(
                "joblib is required for loading the SVM checkpoint. "
                "Install it with `pip install joblib`."
            ) from exc
        return joblib.load(checkpoint_path)

    def train(self) -> dict[str, object]:
        datasets = self.load_train_val_test()
        train_texts, train_labels = self.extract_texts_and_labels(datasets["train"])
        val_texts, val_labels = self.extract_texts_and_labels(datasets["val"])
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])

        self.logger.info("Building TF-IDF features for SVM")
        vectorizer = self._build_vectorizer()
        x_train = vectorizer.fit_transform(train_texts)
        x_val = vectorizer.transform(val_texts)
        x_test = vectorizer.transform(test_texts)

        classifier = self._build_classifier()
        self.logger.info("Training SVM classifier")
        classifier.fit(x_train, train_labels)

        val_predictions = classifier.predict(x_val).tolist()
        test_predictions = classifier.predict(x_test).tolist()
        val_metrics = self.compute_detailed_metrics(val_labels, val_predictions)
        test_metrics = self.compute_detailed_metrics(test_labels, test_predictions)

        checkpoint_path = self._save_checkpoint(vectorizer, classifier)
        prediction_path = self.save_predictions(datasets["test"], test_predictions)
        result_payload = self.build_result_payload(
            mode="train",
            datasets=datasets,
            metrics_by_split={"val": val_metrics, "test": test_metrics},
            extra={
                "best_metric": self.train_config.get("metric_for_best_model", "macro_f1"),
                "best_metric_value": val_metrics.get(
                    self.train_config.get("metric_for_best_model", "macro_f1"),
                    0.0,
                ),
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "model_config": dict(self.model_config),
            },
        )
        results_path = self.save_results(result_payload)

        self.logger.info("Saved SVM checkpoint to %s", checkpoint_path)
        self.logger.info("Saved SVM results to %s", results_path)
        self.logger.info("Saved SVM predictions to %s", prediction_path)
        return {
            "status": "success",
            "checkpoint_path": str(checkpoint_path),
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }

    def test(self) -> dict[str, object]:
        checkpoint_path = self.require_checkpoint()
        checkpoint = self._load_checkpoint(checkpoint_path)
        vectorizer = checkpoint["vectorizer"]
        classifier = checkpoint["classifier"]

        datasets = self.load_test_only()
        test_texts, test_labels = self.extract_texts_and_labels(datasets["test"])
        x_test = vectorizer.transform(test_texts)
        test_predictions = classifier.predict(x_test).tolist()
        test_metrics = self.compute_detailed_metrics(test_labels, test_predictions)

        prediction_path = self.save_predictions(datasets["test"], test_predictions)
        result_payload = self.build_result_payload(
            mode="test",
            datasets=datasets,
            metrics_by_split={"test": test_metrics},
            extra={
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
            },
        )
        results_path = self.save_results(result_payload)
        self.logger.info("Saved SVM test results to %s", results_path)
        self.logger.info("Saved SVM test predictions to %s", prediction_path)
        return {
            "status": "success",
            "results_path": str(results_path),
            "prediction_path": str(prediction_path),
        }
