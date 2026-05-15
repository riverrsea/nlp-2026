from __future__ import annotations

from trainers.base_trainer import BaseTrainer


class BertTrainer(BaseTrainer):
    trainer_name = "bert"

    def train(self) -> dict[str, object]:
        datasets = self.load_train_val_test()
        model_name_or_path = self.config.get("model_name_or_path")
        message = (
            "BERT framework is ready and already accepts `--model_name_or_path`. "
            "Tokenizer loading, training, validation by macro-F1 and checkpoint saving "
            "will be implemented in the next modeling stage."
        )
        note_path = self.save_framework_note(
            mode="train",
            datasets=datasets,
            message=message,
            extra_payload={"model_name_or_path": model_name_or_path},
        )
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}

    def test(self) -> dict[str, object]:
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        model_name_or_path = self.config.get("model_name_or_path")
        message = (
            "BERT test entry has been wired into main.py. "
            "Checkpoint loading and test evaluation will be implemented in the next stage."
        )
        note_path = self.save_framework_note(
            mode="test",
            datasets=datasets,
            message=message,
            extra_payload={
                "checkpoint": str(checkpoint_path),
                "model_name_or_path": model_name_or_path,
            },
        )
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}
