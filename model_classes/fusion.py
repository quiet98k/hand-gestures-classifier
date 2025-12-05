"""Fusion Model that combines GNN and ResNet for multi-modal gesture classification.

This module provides `FusionModel`, which fuses features from a GNN (operating on
hand landmarks) and a ResNet (operating on RGB images) for improved classification.

Usage:
    from model_classes import FusionModel, create_fusion_model
    model = create_fusion_model(num_classes=18)
    # Forward pass with both modalities
    logits = model(image, graph_x, graph_edge_index, graph_batch)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .gnn import GNN, create_gnn
from .resnet import ResNet34, create_resnet34


class FusionModel(nn.Module):
    """Multi-modal fusion model combining GNN and ResNet.

    The model extracts features from both modalities and fuses them
    using late fusion (feature concatenation) followed by an MLP classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    gnn_hidden_channels : int
        Hidden dimension for GNN layers.
    gnn_num_layers : int
        Number of GCN layers in the GNN.
    gnn_dropout : float
        Dropout rate for GNN.
    resnet_dropout : float
        Dropout rate for ResNet.
    resnet_pretrained : bool
        Whether to use pretrained ImageNet weights for ResNet.
    fusion_hidden : int
        Hidden dimension for the fusion MLP classifier.
    fusion_dropout : float
        Dropout rate for fusion classifier.
    freeze_backbones : bool
        If True, freeze GNN and ResNet backbones (train only fusion head).
    gnn_in_channels : int
        Number of input node features for GNN (default 2 for x,y coords).
    resnet_input_channels : int
        Number of input image channels for ResNet (default 3 for RGB).
    """

    def __init__(
        self,
        num_classes: int = 18,
        gnn_hidden_channels: int = 64,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.1,
        resnet_dropout: float = 0.2,
        resnet_pretrained: bool = True,
        fusion_hidden: int = 256,
        fusion_dropout: float = 0.3,
        freeze_backbones: bool = False,
        gnn_in_channels: int = 2,
        resnet_input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.freeze_backbones = freeze_backbones

        # GNN backbone for landmark graphs
        self.gnn = create_gnn(
            in_channels=gnn_in_channels,
            num_classes=num_classes,  # Not used, we use get_features()
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
        )

        # ResNet backbone for RGB images
        self.resnet = create_resnet34(
            input_channels=resnet_input_channels,
            num_classes=num_classes,  # Not used, we use get_features()
            dropout=resnet_dropout,
            pretrained_backbone=resnet_pretrained,
        )

        # Freeze backbones if requested
        if freeze_backbones:
            for param in self.gnn.parameters():
                param.requires_grad = False
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Calculate combined feature dimension
        gnn_feat_dim = self.gnn.feature_dim  # hidden_channels
        resnet_feat_dim = self.resnet.feature_dim  # 512
        combined_dim = gnn_feat_dim + resnet_feat_dim

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

        # Store config for serialization
        self.config = {
            "num_classes": num_classes,
            "gnn_hidden_channels": gnn_hidden_channels,
            "gnn_num_layers": gnn_num_layers,
            "gnn_dropout": gnn_dropout,
            "resnet_dropout": resnet_dropout,
            "resnet_pretrained": resnet_pretrained,
            "fusion_hidden": fusion_hidden,
            "fusion_dropout": fusion_dropout,
            "freeze_backbones": freeze_backbones,
            "gnn_in_channels": gnn_in_channels,
            "resnet_input_channels": resnet_input_channels,
        }

    def forward(
        self,
        image: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with both modalities.

        Parameters
        ----------
        image : torch.Tensor
            RGB images of shape (B, C, H, W).
        graph_x : torch.Tensor
            Node features of shape (num_nodes, in_channels).
        graph_edge_index : torch.Tensor
            Edge index of shape (2, num_edges).
        graph_batch : torch.Tensor
            Batch assignment of shape (num_nodes,).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # Extract features from both modalities
        gnn_features = self.gnn.get_features(graph_x, graph_edge_index, graph_batch)
        resnet_features = self.resnet.get_features(image)

        # Concatenate features
        fused = torch.cat([gnn_features, resnet_features], dim=1)

        # Classify
        return self.classifier(fused)

    def forward_gnn_only(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using only GNN (for comparison/ablation)."""
        return self.gnn(graph_x, graph_edge_index, graph_batch)

    def forward_resnet_only(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass using only ResNet (for comparison/ablation)."""
        return self.resnet(image)

    def unfreeze_backbones(self) -> None:
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.gnn.parameters():
            param.requires_grad = True
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.freeze_backbones = False

    def load_pretrained_backbones(
        self,
        gnn_checkpoint: Optional[str] = None,
        resnet_checkpoint: Optional[str] = None,
    ) -> None:
        """Load pretrained weights for the backbone models.

        Parameters
        ----------
        gnn_checkpoint : Optional[str]
            Path to pretrained GNN checkpoint.
        resnet_checkpoint : Optional[str]
            Path to pretrained ResNet checkpoint.
        """
        if gnn_checkpoint is not None:
            ckpt = torch.load(gnn_checkpoint, map_location="cpu", weights_only=False)
            if "state" in ckpt and "model_state" in ckpt["state"]:
                state_dict = ckpt["state"]["model_state"]
            else:
                state_dict = ckpt
            self.gnn.load_state_dict(state_dict, strict=False)
            print(f"Loaded GNN weights from {gnn_checkpoint}")

        if resnet_checkpoint is not None:
            ckpt = torch.load(resnet_checkpoint, map_location="cpu", weights_only=False)
            if "state" in ckpt and "model_state" in ckpt["state"]:
                state_dict = ckpt["state"]["model_state"]
            else:
                state_dict = ckpt
            self.resnet.load_state_dict(state_dict, strict=False)
            print(f"Loaded ResNet weights from {resnet_checkpoint}")


def create_fusion_model(
    num_classes: int = 18,
    gnn_hidden_channels: int = 64,
    gnn_num_layers: int = 3,
    gnn_dropout: float = 0.1,
    resnet_dropout: float = 0.2,
    resnet_pretrained: bool = True,
    fusion_hidden: int = 256,
    fusion_dropout: float = 0.3,
    freeze_backbones: bool = False,
    gnn_in_channels: int = 2,
    resnet_input_channels: int = 3,
) -> FusionModel:
    """Factory function to create a FusionModel.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    gnn_hidden_channels : int
        Hidden dimension for GNN layers.
    gnn_num_layers : int
        Number of GCN layers in the GNN.
    gnn_dropout : float
        Dropout rate for GNN.
    resnet_dropout : float
        Dropout rate for ResNet.
    resnet_pretrained : bool
        Whether to use pretrained ImageNet weights for ResNet.
    fusion_hidden : int
        Hidden dimension for the fusion MLP classifier.
    fusion_dropout : float
        Dropout rate for fusion classifier.
    freeze_backbones : bool
        If True, freeze GNN and ResNet backbones.
    gnn_in_channels : int
        Number of input node features for GNN.
    resnet_input_channels : int
        Number of input image channels for ResNet.

    Returns
    -------
    FusionModel
        Configured fusion model.
    """
    return FusionModel(
        num_classes=num_classes,
        gnn_hidden_channels=gnn_hidden_channels,
        gnn_num_layers=gnn_num_layers,
        gnn_dropout=gnn_dropout,
        resnet_dropout=resnet_dropout,
        resnet_pretrained=resnet_pretrained,
        fusion_hidden=fusion_hidden,
        fusion_dropout=fusion_dropout,
        freeze_backbones=freeze_backbones,
        gnn_in_channels=gnn_in_channels,
        resnet_input_channels=resnet_input_channels,
    )


if __name__ == "__main__":
    print("=== Sanity Check: FusionModel ===")

    batch_size = 4
    image_size = 128
    num_classes = 18
    num_nodes_per_graph = 21

    # Create model
    model = create_fusion_model(
        num_classes=num_classes,
        resnet_pretrained=False,  # Skip pretrained for quick test
    )
    model.eval()

    # Create dummy inputs
    images = torch.rand(batch_size, 3, image_size, image_size)
    graph_x = torch.rand(batch_size * num_nodes_per_graph, 2)
    # Simple edge_index for testing
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph_batch = torch.repeat_interleave(
        torch.arange(batch_size), num_nodes_per_graph
    )

    # Forward pass
    with torch.no_grad():
        logits = model(images, graph_x, edge_index, graph_batch)

    print(f"Input image shape: {images.shape}")
    print(f"Input graph_x shape: {graph_x.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("=== Sanity Check Complete ===")
