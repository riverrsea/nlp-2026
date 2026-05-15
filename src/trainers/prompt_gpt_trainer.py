from __future__ import annotations

from trainers.base_trainer import BaseTrainer


class PromptGPTTrainer(BaseTrainer):
    trainer_name = "prompt_gpt"

    def train(self) -> dict[str, object]:
        datasets = self.load_train_val_test()
        message = (
            "Prompt-GPT trainer has been registered in the framework. "
            "API invocation, prompt construction and prediction parsing will be implemented in a later stage."
        )
        note_path = self.save_framework_note(mode="train", datasets=datasets, message=message)
        self.logger.warning(message)
        self.logger.info("Saved framework note to %s", note_path)
        return {"status": "framework_ready", "note_path": str(note_path)}

    def test(self) -> dict[str, object]:
        checkpoint_path = self.require_checkpoint()
        datasets = self.load_test_only()
        message = (
            "Prompt-GPT test entry has been wired into main.py. "
            "Model calling and output parsing will be implemented in a later stage."
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
