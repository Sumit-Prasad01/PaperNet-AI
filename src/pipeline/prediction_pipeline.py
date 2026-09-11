import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from src.config import ConfigurationManager
from src.models import AdvancedCoraGAT
from utils.custom_exception import CustomException
from utils.helpers import get_device
from utils.logger import logger

CORA_CLASS_LABELS: Dict[int, str] = {
    0: "Rule_Learning",
    1: "Neural_Networks",
    2: "Genetic_Algorithms",
    3: "Probabilistic_Methods",
    4: "Case_Based",
    5: "Reinforcement_Learning",
    6: "Theory",
}


class PredictionPipeline:
    """Inference pipeline for topic classification on research paper graphs."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        onnx_model_path: Optional[Union[str, Path]] = None,
        device_str: str = "auto",
    ):
        try:
            config_manager = ConfigurationManager()
            trainer_config = config_manager.get_model_trainer_config()
            eval_config = config_manager.get_model_evaluation_config()
            self.arch_config = config_manager.get_model_architecture_config()

            self.model_path = (
                Path(model_path) if model_path else trainer_config.model_path
            )
            self.onnx_model_path = (
                Path(onnx_model_path) if onnx_model_path else eval_config.onnx_model_path
            )
            self.device = get_device(device_str)

            self.model: Optional[torch.nn.Module] = None
            self.ort_session: Optional[ort.InferenceSession] = None
        except Exception as e:
            logger.error("Failed to initialize PredictionPipeline")
            raise CustomException(e, sys)

    def load_pytorch_model(self) -> torch.nn.Module:
        """Loads the trained PyTorch GNN model checkpoint."""
        try:
            if self.model is None:
                logger.info(f"Loading PyTorch model from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                model = AdvancedCoraGAT(
                    input_dim=checkpoint["input_dim"],
                    hidden_dim=checkpoint["hidden_dim"],
                    num_classes=checkpoint["num_classes"],
                    num_layers=checkpoint["num_layers"],
                    heads=checkpoint["heads"],
                    dropout=checkpoint.get("dropout", self.arch_config.dropout),
                    attention_dropout=checkpoint.get(
                        "attention_dropout", self.arch_config.attention_dropout
                    ),
                    classifier_dropout_1=self.arch_config.classifier_dropout_1,
                    classifier_dropout_2=self.arch_config.classifier_dropout_2,
                    jk_mode=self.arch_config.jk_mode,
                ).to(self.device)

                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.model = model
                logger.info("PyTorch model loaded successfully.")
            return self.model
        except Exception as e:
            logger.error("Failed to load PyTorch model")
            raise CustomException(e, sys)

    def load_onnx_model(self) -> ort.InferenceSession:
        """Initializes ONNX Runtime inference session."""
        try:
            if self.ort_session is None:
                logger.info(f"Loading ONNX model session from: {self.onnx_model_path}")
                self.ort_session = ort.InferenceSession(str(self.onnx_model_path))
                logger.info("ONNX Runtime session initialized.")
            return self.ort_session
        except Exception as e:
            logger.error("Failed to load ONNX model session")
            raise CustomException(e, sys)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs graph node classification using PyTorch model.

        Args:
            x: Node features tensor [num_nodes, feature_dim].
            edge_index: Graph connectivity tensor [2, num_edges].
            node_indices: Optional list of target node indices to predict for.

        Returns:
            List[Dict[str, Any]]: List of predicted class ids, labels, and confidences.
        """
        try:
            model = self.load_pytorch_model()
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)

            logits = model(x, edge_index)
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            results = []
            targets = node_indices if node_indices is not None else range(x.size(0))

            for idx in targets:
                class_id = int(predicted_classes[idx].item())
                confidence = float(probabilities[idx, class_id].item())
                label = CORA_CLASS_LABELS.get(class_id, f"Class_{class_id}")
                results.append({
                    "node_index": int(idx),
                    "predicted_class_id": class_id,
                    "topic_label": label,
                    "confidence": round(confidence, 4),
                })
            return results
        except Exception as e:
            logger.error("Failed during PyTorch prediction")
            raise CustomException(e, sys)

    def predict_onnx(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        node_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs inference using ONNX Runtime.

        Args:
            x: Node feature array [num_nodes, feature_dim].
            edge_index: Edge index array [2, num_edges].
            node_indices: Optional target node indices to return predictions for.

        Returns:
            List[Dict[str, Any]]: Prediction results.
        """
        try:
            session = self.load_onnx_model()
            inputs = {
                "node_features": x.astype(np.float32),
                "edge_index": edge_index.astype(np.int64),
            }
            outputs = session.run(None, inputs)
            logits = outputs[0]

            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            predicted_classes = np.argmax(probabilities, axis=-1)

            results = []
            targets = node_indices if node_indices is not None else range(len(x))

            for idx in targets:
                class_id = int(predicted_classes[idx])
                confidence = float(probabilities[idx, class_id])
                label = CORA_CLASS_LABELS.get(class_id, f"Class_{class_id}")
                results.append({
                    "node_index": int(idx),
                    "predicted_class_id": class_id,
                    "topic_label": label,
                    "confidence": round(confidence, 4),
                })
            return results
        except Exception as e:
            logger.error("Failed during ONNX prediction")
            raise CustomException(e, sys)
