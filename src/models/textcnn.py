from __future__ import annotations


try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


if nn is not None:
    class TextCNN(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            num_classes: int,
            num_filters: int = 128,
            kernel_sizes: tuple[int, ...] = (3, 4, 5),
            dropout: float = 0.5,
            padding_idx: int = 0,
            embedding_weights=None,
            freeze_embeddings: bool = False,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
            if embedding_weights is not None:
                weight_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
                self.embedding.weight.data.copy_(weight_tensor)
                self.embedding.weight.requires_grad = not freeze_embeddings

            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=num_filters,
                        kernel_size=kernel_size,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            pooled_outputs = []
            for conv in self.convs:
                conv_output = torch.relu(conv(embedded))
                pooled_output = torch.max(conv_output, dim=2).values
                pooled_outputs.append(pooled_output)
            features = torch.cat(pooled_outputs, dim=1)
            features = self.dropout(features)
            return self.classifier(features)
else:
    class TextCNN:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "torch is required to build the TextCNN model. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from TORCH_IMPORT_ERROR
