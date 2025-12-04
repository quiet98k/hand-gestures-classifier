"""ResNet-34 for hand gesture classification.

This module provides `ResNet34`, a model based on the ResNet-34 architecture
for classifying hand gesture images from the `RGBImageDataset`.

Usage:
    from model_classes import ResNet34, create_resnet34
    model = create_resnet34(input_channels=3, num_classes=18)
    logits = model(images)  # images: (B, 3, H, W)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34.
    
    Two 3x3 conv layers with skip connection.
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    """ResNet-34 for gesture classification from image crops.

    Expects input of shape (B, C, H, W).

    Parameters
    ----------
    input_channels : int
        Number of image channels (default 3).
    num_classes : int
        Number of gesture classes.
    dropout : float
        Dropout probability before the final classifier.
    pretrained_backbone : bool
        If True, initialize from torchvision pretrained weights (requires input_channels=3).
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 18,
        dropout: float = 0.1,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pretrained_backbone = pretrained_backbone

        if pretrained_backbone:
            # Use torchvision pretrained ResNet-34
            try:
                from torchvision.models import resnet34, ResNet34_Weights
                backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            except ImportError:
                from torchvision.models import resnet34
                backbone = resnet34(pretrained=True)
            
            # Modify first conv if input_channels != 3
            if input_channels != 3:
                old_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Initialize new conv with mean of pretrained weights
                with torch.no_grad():
                    backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).expand_as(backbone.conv1.weight)
            
            # Extract backbone layers (remove fc)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        else:
            # Build from scratch
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # ResNet-34 layer configuration: [3, 4, 6, 3]
            self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
            self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
            self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
            self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

            # Initialize weights
            self._init_weights()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W); got shape {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected channel dimension {self.input_channels}; got {x.size(1)}"
            )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_resnet34(
    input_channels: int = 3,
    num_classes: int = 18,
    dropout: float = 0.1,
    pretrained_backbone: bool = False,
) -> ResNet34:
    """Factory to create a ResNet-34 model.

    Parameters
    ----------
    input_channels : int
        Number of image channels.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability before classifier.
    pretrained_backbone : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    ResNet34
        Configured ResNet-34 model.
    """
    return ResNet34(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=dropout,
        pretrained_backbone=pretrained_backbone,
    )


# Run with: uv run model_classes/resnet.py
if __name__ == "__main__":
    print("=== Sanity Check: ResNet34 ===")

    batch_size = 4
    input_channels = 3
    height = 64
    width = 64
    num_classes = 18

    # Test without pretrained
    print("\nTesting ResNet-34 (from scratch)...")
    model = create_resnet34(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=0.1,
        pretrained_backbone=False,
    )
    model.eval()

    x = torch.rand(batch_size, input_channels, height, width)
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test with pretrained (optional)
    try:
        print("\nTesting ResNet-34 (pretrained)...")
        model_pt = create_resnet34(
            input_channels=3,
            num_classes=num_classes,
            dropout=0.1,
            pretrained_backbone=True,
        )
        model_pt.eval()
        with torch.no_grad():
            logits_pt = model_pt(x)
        print(f"Pretrained output shape: {logits_pt.shape}")
    except Exception as e:
        print(f"Skipping pretrained test: {e}")

    print("\n=== Sanity Check Complete ===")
