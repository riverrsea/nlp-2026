from __future__ import annotations

import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset as TorchDataset  # type: ignore
except Exception:
    torch = None

    class TorchDataset:  # type: ignore
        """Fallback base class when torch is unavailable."""

        pass


def tokenize_with_jieba(text: str) -> list[str]:
    try:
        import jieba  # type: ignore
    except Exception:
        warnings.warn(
            "jieba is not installed. Falling back to character-level tokenization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return [char for char in text if not char.isspace()]

    return [token.strip() for token in jieba.cut(text) if token.strip()]


def build_vocab(
    texts: Sequence[str],
    tokenizer: Callable[[str], list[str]] = tokenize_with_jieba,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> dict[str, int]:
    token_counter: Counter[str] = Counter()
    for text in texts:
        token_counter.update(tokenizer(text))

    sorted_items = sorted(token_counter.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = {pad_token: 0, unk_token: 1}

    for token, count in sorted_items:
        if count < min_freq:
            continue
        if token in vocabulary:
            continue
        if max_vocab_size is not None and len(vocabulary) >= max_vocab_size:
            break
        vocabulary[token] = len(vocabulary)

    return vocabulary


def tokens_to_ids(
    tokens: Sequence[str],
    vocabulary: Mapping[str, int],
    unk_token: str = "<UNK>",
) -> list[int]:
    unk_id = vocabulary[unk_token]
    return [vocabulary.get(token, unk_id) for token in tokens]


def pad_or_truncate(
    token_ids: Sequence[int],
    max_length: int,
    pad_id: int = 0,
) -> tuple[list[int], int]:
    truncated = list(token_ids[:max_length])
    original_length = min(len(token_ids), max_length)
    if len(truncated) < max_length:
        truncated.extend([pad_id] * (max_length - len(truncated)))
    return truncated, original_length


def prepare_textcnn_inputs(
    texts: Sequence[str],
    tokenizer: Callable[[str], list[str]] = tokenize_with_jieba,
    vocabulary: Mapping[str, int] | None = None,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    max_length: int = 256,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> dict[str, object]:
    vocab = (
        dict(vocabulary)
        if vocabulary is not None
        else build_vocab(
            texts=texts,
            tokenizer=tokenizer,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            pad_token=pad_token,
            unk_token=unk_token,
        )
    )

    input_ids: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        tokens = tokenizer(text)
        token_ids = tokens_to_ids(tokens, vocab, unk_token=unk_token)
        padded_ids, sequence_length = pad_or_truncate(
            token_ids,
            max_length=max_length,
            pad_id=vocab[pad_token],
        )
        input_ids.append(padded_ids)
        lengths.append(sequence_length)

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "vocab": vocab,
    }


def build_random_embedding_matrix(
    vocabulary: Mapping[str, int],
    embedding_dim: int,
    seed: int = 42,
) -> object:
    rng = random.Random(seed)
    matrix = [
        [rng.uniform(-0.25, 0.25) for _ in range(embedding_dim)]
        for _ in range(len(vocabulary))
    ]
    matrix[0] = [0.0] * embedding_dim
    if np is not None:
        return np.asarray(matrix, dtype="float32")
    return matrix


def load_pretrained_word_vectors(
    embedding_path: str | Path,
    vocabulary: Mapping[str, int],
    embedding_dim: int,
    pad_token: str = "<PAD>",
    encoding: str = "utf-8",
) -> object:
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained word vector file not found: {path}. "
            "Please provide a valid embedding file path."
        )

    if np is not None:
        embedding_matrix = build_random_embedding_matrix(
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
        )
        assert isinstance(embedding_matrix, np.ndarray)
    else:
        embedding_matrix = build_random_embedding_matrix(
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
        )

    with path.open("r", encoding=encoding, errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) <= embedding_dim:
                continue
            token = parts[0]
            if token not in vocabulary:
                continue
            vector_values = parts[-embedding_dim:]
            try:
                vector = [float(value) for value in vector_values]
            except ValueError:
                continue
            embedding_matrix[vocabulary[token]] = vector

    if pad_token in vocabulary:
        embedding_matrix[vocabulary[pad_token]] = [0.0] * embedding_dim

    if np is not None and not isinstance(embedding_matrix, np.ndarray):
        embedding_matrix = np.asarray(embedding_matrix, dtype="float32")
    return embedding_matrix


def load_bert_tokenizer(
    model_name_or_path: str = "bert-base-chinese",
    cache_dir: str | Path | None = None,
    use_fast: bool = True,
):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise ImportError(
            "transformers is required for BERT tokenizer preparation. "
            "Install it with `pip install transformers`."
        ) from exc

    tokenizer_kwargs = {"use_fast": use_fast}
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = str(cache_dir)
    return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)


def prepare_bert_inputs(
    texts: Sequence[str],
    tokenizer,
    max_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str | None = None,
) -> Mapping[str, object]:
    return tokenizer(
        list(texts),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )


class TextCNNDataset(TorchDataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Callable[[str], list[str]] = tokenize_with_jieba,
        vocabulary: Mapping[str, int] | None = None,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        max_length: int = 256,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ) -> None:
        prepared = prepare_textcnn_inputs(
            texts=texts,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            max_length=max_length,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        self.input_ids = prepared["input_ids"]
        self.lengths = prepared["lengths"]
        self.vocab = prepared["vocab"]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        item = {
            "input_ids": self.input_ids[index],
            "length": self.lengths[index],
            "label": self.labels[index],
        }
        if torch is not None:
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "length": torch.tensor(item["length"], dtype=torch.long),
                "label": torch.tensor(item["label"], dtype=torch.long),
            }
        return item


class BertTextDataset(TorchDataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        self.encodings = prepare_bert_inputs(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["label"] = self.labels[index]

        if torch is not None:
            tensor_item = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in item.items()
            }
            return tensor_item
        return item
