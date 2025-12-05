"""Graph Neural Network for hand gesture classification.

This module provides `GNN`, a model that operates on hand landmark graphs
produced by `LandmarksGraphDataset`. It uses Graph Convolutional layers
followed by global pooling and an MLP classifier.

Usage:
    from model_classes import GNN, create_gnn
    model = create_gnn(in_channels=2, num_classes=18)
    # Pass PyG Data batch
    logits = model(batch.x, batch.edge_index, batch.batch)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GNN(nn.Module):
    """GNN for gesture classification from hand landmark graphs.

    The model applies GCN layers to node features, uses global mean pooling
    to get a graph-level representation, then an MLP classifier.

    Parameters
    ----------
    in_channels : int
        Number of input node features (e.g., 2 for x/y coords).
    num_classes : int
        Number of gesture classes.
    hidden_channels : int
        Hidden dimension for GCN layers.
    num_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability.
    activation : str
        'relu' or 'gelu'.
    mlp_hidden : Optional[int]
        Hidden dimension for classifier MLP. If None, uses hidden_channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 18,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        mlp_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GNN. "
                "Install it with: pip install torch_geometric"
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_name = activation.lower()
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else hidden_channels

        # Activation function
        self.act = nn.ReLU() if self.activation_name == "relu" else nn.GELU()

        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, self.mlp_hidden),
            nn.BatchNorm1d(self.mlp_hidden),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(self.mlp_hidden, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge index of shape (2, num_edges).
        batch : Optional[torch.Tensor]
            Batch assignment vector of shape (num_nodes,).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN layers
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)

    def get_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before the classifier head.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge index of shape (2, num_edges).
        batch : Optional[torch.Tensor]
            Batch assignment vector of shape (num_nodes,).

        Returns
        -------
        torch.Tensor
            Feature embeddings of shape (batch_size, hidden_channels).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN layers
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        return x

    @property
    def feature_dim(self) -> int:
        """Return the dimension of the feature embeddings."""
        return self.hidden_channels


def create_gnn(
    in_channels: int = 2,
    num_classes: int = 18,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    activation: str = "relu",
    mlp_hidden: Optional[int] = None,
) -> GNN:
    """Factory to create a GNN model."""
    return GNN(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        mlp_hidden=mlp_hidden,
    )


# Run with: uv run model_classes/gnn.py
if __name__ == "__main__":
    print("=== Sanity Check: GNN ===")

    try:
        from torch_geometric.data import Data, Batch

        num_nodes = 21
        in_channels = 2
        num_classes = 18

        x = torch.randn(num_nodes, in_channels)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([0]))
        print(f"Single graph: {data}")

        model = create_gnn(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=64,
            num_layers=3,
            dropout=0.1,
        )
        model.eval()

        with torch.no_grad():
            logits = model(data.x, data.edge_index, batch=None)
        print(f"Single graph output shape: {logits.shape}")

        batch = Batch.from_data_list([data, data, data])
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.batch)
        print(f"Batched (3 graphs) output shape: {logits.shape}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        print("=== Sanity Check Complete ===")

    except ImportError as e:
        print(f"Cannot run sanity check: {e}")
