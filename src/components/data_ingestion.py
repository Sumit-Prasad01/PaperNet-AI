import random
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, to_networkx

from src.entity import DataIngestionConfig
from utils.custom_exception import CustomException
from utils.logger import logger


class DataIngestion:
    """Handles loading and statistical profiling of graph dataset (Cora)."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self) -> Tuple[Planetoid, Data]:
        """Loads the Planetoid dataset and logs summary graph metrics.

        Returns:
            Tuple[Planetoid, Data]: The dataset container and the primary graph Data object.
        """
        try:
            logger.info(
                f"Loading {self.config.dataset_name} dataset from root: {self.config.root_dir}"
            )
            dataset = Planetoid(
                root=str(self.config.root_dir),
                name=self.config.dataset_name,
            )
            data = dataset[0]

            logger.info(f"Dataset successfully loaded. Number of graphs: {len(dataset)}")
            logger.info(f"Nodes          : {data.num_nodes}")
            logger.info(f"Edges          : {data.num_edges}")
            logger.info(f"Features       : {data.num_node_features}")
            logger.info(f"Classes        : {dataset.num_classes}")
            logger.info(f"Train nodes    : {data.train_mask.sum().item()}")
            logger.info(f"Val nodes      : {data.val_mask.sum().item()}")
            logger.info(f"Test nodes     : {data.test_mask.sum().item()}")

            # Degree statistics
            row, _ = data.edge_index
            node_degree = degree(row, num_nodes=data.num_nodes)
            logger.info(
                f"Node Degree - Avg: {node_degree.mean().item():.2f} | "
                f"Min: {node_degree.min().item():.0f} | "
                f"Max: {node_degree.max().item():.0f}"
            )

            # Class distribution
            class_counts = (
                pd.Series(data.y.cpu().numpy()).value_counts().sort_index().to_dict()
            )
            logger.info(f"Class distribution across nodes: {class_counts}")

            return dataset, data
        except Exception as e:
            logger.error("Failed to load or profile graph dataset")
            raise CustomException(e, sys)

    def visualize_graph_sample(
        self,
        data: Data,
        output_path: Path,
        num_nodes: int = 100,
        seed: int = 12345,
    ) -> None:
        """Visualizes a sampled subgraph colored by paper topic class.

        Args:
            data: PyG Data object.
            output_path: Destination image path.
            num_nodes: Number of sampled nodes.
            seed: Random seed for sampling and spring layout.
        """
        try:
            logger.info(f"Generating subgraph visualization with {num_nodes} nodes...")
            random.seed(seed)
            G = to_networkx(data, to_undirected=True)
            sampled_nodes = random.sample(list(G.nodes()), min(num_nodes, len(G.nodes())))
            sample_graph = G.subgraph(sampled_nodes)

            node_colors = [data.y[node].item() for node in sample_graph.nodes()]
            pos = nx.spring_layout(sample_graph, seed=seed)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(12, 10))
            nx.draw(
                sample_graph,
                pos,
                node_color=node_colors,
                cmap=plt.cm.jet,
                node_size=100,
                alpha=0.9,
                with_labels=False,
            )
            plt.title(f"{num_nodes} Random Research Papers Colored by Topic")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Subgraph visualization saved at: {output_path}")
        except Exception as e:
            logger.error("Failed to generate subgraph visualization")
            raise CustomException(e, sys)
