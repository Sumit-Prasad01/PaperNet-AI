import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, JumpingKnowledge

from utils.custom_exception import CustomException
from utils.logger import logger


class GATv2Block(nn.Module):
    """A residual Graph Attention Network v2 (GATv2) layer block with LayerNorm, GELU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.35,
        attention_dropout: float = 0.20,
    ):
        super().__init__()
        try:
            assert (
                out_channels % heads == 0
            ), f"out_channels ({out_channels}) must be divisible by heads ({heads})"

            self.conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels // heads,
                heads=heads,
                concat=True,
                dropout=attention_dropout,
                add_self_loops=True,
            )
            self.norm = nn.LayerNorm(out_channels)
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Linear(in_channels, out_channels, bias=False)
            )
            self.dropout = nn.Dropout(dropout)
        except Exception as e:
            logger.error("Failed to initialize GATv2Block")
            raise CustomException(e, sys)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of GATv2Block."""
        try:
            residual = self.residual(x)
            x = self.conv(x, edge_index)
            x = self.norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + residual
            return x
        except Exception as e:
            logger.error("Error during GATv2Block forward pass")
            raise CustomException(e, sys)


class AdvancedCoraGAT(nn.Module):
    """Deep GATv2 architecture with JumpingKnowledge and MLP classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.35,
        attention_dropout: float = 0.20,
        classifier_dropout_1: float = 0.40,
        classifier_dropout_2: float = 0.25,
        jk_mode: str = "cat",
    ):
        super().__init__()
        try:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.heads = heads

            # Initial feature projection
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # GATv2 layers
            self.gnn_layers = nn.ModuleList([
                GATv2Block(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ])

            # Jumping Knowledge aggregation
            self.jk = JumpingKnowledge(mode=jk_mode)
            jk_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim

            # Multi-layer classification head
            self.classifier = nn.Sequential(
                nn.Linear(jk_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout_1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(classifier_dropout_2),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        except Exception as e:
            logger.error("Failed to initialize AdvancedCoraGAT")
            raise CustomException(e, sys)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through input projection, GNN layers, JK, and classifier."""
        try:
            # Initial feature projection
            x = self.input_projection(x)

            representations = []
            # GNN layers
            for layer in self.gnn_layers:
                x = layer(x, edge_index)
                representations.append(x)

            # Jumping Knowledge
            x = self.jk(representations)

            # Classification
            logits = self.classifier(x)
            return logits
        except Exception as e:
            logger.error("Error during AdvancedCoraGAT forward pass")
            raise CustomException(e, sys)
