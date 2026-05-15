from __future__ import annotations

from trainers.base_trainer import BaseTrainer


class PromptBertTrainer(BaseTrainer):
    trainer_name = "prompt_bert"

    def train(self) -> dict[str, object]:
        datasets = self.load_train_val_test()
        message = (
            "Prompt-BERT trainer has been registered in the framework. "
            "Prompt template construction and downstream training logic will be implemented in a later stage."
        )
        note_path = self.save_framework_note(mode="train", datasets=datasets, message=message)
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}

    def test(self) -> dict[str, object]:
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        message = (
            "Prompt-BERT test entry has been wired into main.py. "
            "Checkpoint loading and prompt-based evaluation will be implemented in a later stage."
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
