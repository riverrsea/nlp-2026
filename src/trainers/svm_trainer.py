from __future__ import annotations

from trainers.base_trainer import BaseTrainer


class SVMTrainer(BaseTrainer):
    trainer_name = "svm"

    def train(self) -> dict[str, object]:
        datasets = self.load_train_val_test()
        message = (
            "SVM trainer has been registered and the project framework is ready. "
            "The actual TF-IDF + SVM training logic will be implemented in the next modeling stage."
        )
        note_path = self.save_framework_note(mode="train", datasets=datasets, message=message)
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}

    def test(self) -> dict[str, object]:
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        message = (
            "SVM test entry has been wired into main.py. "
            "Checkpoint loading and evaluation will be implemented in the next stage."
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
