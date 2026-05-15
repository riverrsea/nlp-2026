from __future__ import annotations

from pathlib import Path

from trainers.base_trainer import BaseTrainer


class TextCNNTrainer(BaseTrainer):
    trainer_name = "textcnn"

    def _validate_pretrained_path_if_needed(self) -> Path | None:
        embedding_type = str(self.config.get("embedding_type", "random")).lower()
        if embedding_type != "pretrained":
            return None

        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path is None:
            raise ValueError(
                "TextCNN with pretrained embeddings requires `--pretrained_path` "
                "or `pretrained_path` in the YAML config."
            )

        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained word vector file not found: {path}. "
                "Please provide a valid path through `--pretrained_path`."
            )
        return path

    def train(self) -> dict[str, object]:
        pretrained_path = self._validate_pretrained_path_if_needed()
        datasets = self.load_train_val_test()
        embedding_type = str(self.config.get("embedding_type", "random")).lower()
        message = (
            f"TextCNN framework is ready with embedding_type={embedding_type}. "
            "Model definition, optimizer setup, validation by macro-F1 and checkpoint saving "
            "will be implemented in the next modeling stage."
        )
        extra = {"embedding_type": embedding_type}
        if pretrained_path is not None:
            extra["pretrained_path"] = str(pretrained_path)
        note_path = self.save_framework_note(
            mode="train",
            datasets=datasets,
            message=message,
            extra_payload=extra,
        )
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}

    def test(self) -> dict[str, object]:
        self._validate_pretrained_path_if_needed()
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        message = (
            "TextCNN test entry has been wired into main.py. "
            "Checkpoint loading, batching and evaluation will be implemented in the next stage."
        )
        note_path = self.save_framework_note(
            mode="test",
            datasets=datasets,
            message=message,
            extra_payload={"checkpoint": str(checkpoint_path)},
        )
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}
