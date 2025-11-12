from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Minimal MLP baseline for gesture classification from flattened landmarks.

    Expects input of shape (B, D) where D = num_landmarks * coords_per_landmark
    (already flattened by the data pipeline). No internal reshaping.

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., 21 landmarks * 2 coords = 42).
    num_classes : int
        Number of gesture classes.
    hidden_dims : tuple[int, ...]
        Sizes of hidden layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    activation : str
        'relu' or 'gelu'.
    batchnorm : bool
        If True, insert BatchNorm1d after Linear and before activation.
    """

    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 18,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation_name = activation
        self.batchnorm = batchnorm

        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        layers: list[nn.Module] = []

        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect already flattened input (B, D)
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D (B, D); got shape {tuple(x.shape)}")
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}; got {x.size(1)}"
            )
        z = self.backbone(x)
        return self.head(z)


def create_mlp_baseline(
    input_dim: int = 42,
    num_classes: int = 18,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.1,
    activation: str = "relu",
    batchnorm: bool = False,
) -> MLPBaseline:
    """Factory to create an MLPBaseline with an explicit flattened input size."""
    return MLPBaseline(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        batchnorm=batchnorm,
    )


if __name__ == "__main__":
    # Simple sanity run
    batch_size = 8
    input_dim = 42  # e.g., 21 * 2
    num_classes = 18

    model = create_mlp_baseline(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(128, 64),
        dropout=0.1,
        activation="relu",
        batchnorm=False,
    )
    model.eval()

    # Dummy normalized landmark batch (already bbox-normalized in dataset)
    x = torch.rand(batch_size, input_dim)
    with torch.no_grad():
        logits = model(x)
    print(model)
    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Pred class ids:", logits.argmax(dim=1))
