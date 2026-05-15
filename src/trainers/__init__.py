from __future__ import annotations

import importlib


TRAINER_REGISTRY = {
    "svm": ("trainers.svm_trainer", "SVMTrainer"),
    "textcnn": ("trainers.textcnn_trainer", "TextCNNTrainer"),
    "bert": ("trainers.bert_trainer", "BertTrainer"),
    "prompt_bert": ("trainers.prompt_bert_trainer", "PromptBertTrainer"),
    "prompt_gpt": ("trainers.prompt_gpt_trainer", "PromptGPTTrainer"),
}


def get_trainer_class(model_name: str):
    if model_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Supported values are: {sorted(TRAINER_REGISTRY)}"
        )

    module_name, class_name = TRAINER_REGISTRY[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
