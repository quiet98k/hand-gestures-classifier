from __future__ import annotations

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """Simple CNN baseline for gesture classification from image crops.

    Expects input of shape (B, C, H, W).

    Parameters
    ----------
    input_channels : int
        Number of image channels (default 3).
    num_classes : int
        Number of gesture classes.
    conv_channels : tuple[int, ...]
        Number of output channels for successive Conv2d layers.
    kernel_size : int
        Convolution kernel size.
    pool_every : int
        Insert a MaxPool2d(2) after this many conv blocks.
    dropout : float
        Dropout probability (applied as Dropout2d after conv blocks).
    activation : str
        'relu' or 'gelu'.
    batchnorm : bool
        If True, insert BatchNorm2d after Conv2d and before activation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 18,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        pool_every: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_every = pool_every
        self.dropout = dropout
        self.activation_name = activation
        self.batchnorm = batchnorm

        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        layers: list[nn.Module] = []

        in_ch = input_channels
        for i, out_ch in enumerate(conv_channels):
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act())
            # Add pooling every `pool_every` conv blocks
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        # Use adaptive pooling to decouple head from input resolution.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect image input (B, C, H, W)
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W); got shape {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected channel dimension {self.input_channels}; got {x.size(1)}"
            )
        z = self.backbone(x)
        z = self.global_pool(z)  # (B, C_last, 1, 1)
        z = torch.flatten(z, 1)  # (B, C_last)
        return self.head(z)


def create_cnn_baseline(
    input_channels: int = 3,
    num_classes: int = 18,
    conv_channels: tuple[int, ...] = (32, 64),
    kernel_size: int = 3,
    pool_every: int = 2,
    dropout: float = 0.1,
    activation: str = "relu",
    batchnorm: bool = False,
) -> CNNBaseline:
    """Factory to create a CNNBaseline with sensible defaults for small images."""
    return CNNBaseline(
        input_channels=input_channels,
        num_classes=num_classes,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        pool_every=pool_every,
        dropout=dropout,
        activation=activation,
        batchnorm=batchnorm,
    )


if __name__ == "__main__":
    # Simple sanity run
    batch_size = 8
    input_channels = 3
    height = 64
    width = 64
    num_classes = 18

    model = create_cnn_baseline(
        input_channels=input_channels,
        num_classes=num_classes,
        conv_channels=(32, 64),
        dropout=0.1,
        activation="relu",
        batchnorm=False,
    )
    model.eval()

    # Dummy image batch
    x = torch.rand(batch_size, input_channels, height, width)
    with torch.no_grad():
        logits = model(x)
    print(model)
    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Pred class ids:", logits.argmax(dim=1))
