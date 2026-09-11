import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.data import Data

from src.entity import ModelEvaluationConfig
from utils.custom_exception import CustomException
from utils.helpers import save_json
from utils.logger import logger


class ModelEvaluation:
    """Evaluates the trained model on test split, generates diagnostic plots, and exports to ONNX."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(
        self,
        model: torch.nn.Module,
        data: Data,
        history: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Runs complete evaluation, saves metrics and plots, and exports to ONNX.

        Args:
            model: Trained PyTorch GNN model.
            data: Graph Data object.
            history: Training history metrics dictionary.

        Returns:
            Dict[str, Any]: Test evaluation metrics.
        """
        try:
            logger.info("Evaluating model on test mask...")
            model.eval()

            # Ensure data is on the same device as model
            device = next(model.parameters()).device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            with torch.no_grad():
                logits = model(x, edge_index)
                predictions = logits.argmax(dim=1)

            test_mask = data.test_mask.cpu()
            y_test = data.y[test_mask].cpu().numpy()
            pred_test = predictions[test_mask].cpu().numpy()

            test_accuracy = float(accuracy_score(y_test, pred_test))
            test_macro_f1 = float(f1_score(y_test, pred_test, average="macro", zero_division=0))

            logger.info(f"Test Accuracy: {test_accuracy:.4f} | Test Macro-F1: {test_macro_f1:.4f}")

            # Classification report
            report_str = classification_report(y_test, pred_test, digits=4)
            logger.info(f"\nClassification Report:\n{report_str}")

            with open(self.config.classification_report_file, "w", encoding="utf-8") as f:
                f.write(report_str)
            logger.info(f"Saved classification report at: {self.config.classification_report_file}")

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, pred_test)
            self._save_confusion_matrix(cm)

            # Training Curves Plot
            if history:
                self._save_training_curves(history)

            # Save metrics JSON
            metrics = {
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "best_epoch": history.get("best_epoch") if history else None,
                "best_val_acc": history.get("best_val_acc") if history else None,
                "best_val_f1": history.get("best_val_f1") if history else None,
            }
            save_json(self.config.metrics_file, metrics)

            # Export to ONNX
            self._export_to_onnx(model, x, edge_index)

            return metrics
        except Exception as e:
            logger.error("Failed during model evaluation and export")
            raise CustomException(e, sys)

    def _save_confusion_matrix(self, cm: np.ndarray) -> None:
        """Generates and saves a confusion matrix heatmap plot."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Advanced GATv2 - Cora Confusion Matrix")
            plt.tight_layout()
            plt.savefig(self.config.confusion_matrix_plot, dpi=300)
            plt.close()
            logger.info(f"Saved confusion matrix plot at: {self.config.confusion_matrix_plot}")
        except Exception as e:
            logger.error("Failed to generate confusion matrix plot")
            raise CustomException(e, sys)

    def _save_training_curves(self, history: Dict[str, Any]) -> None:
        """Generates and saves multi-panel training diagnostic curves."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Accuracy curve
            axes[0].plot(history.get("train_accs", []), label="Train Accuracy", color="#1f77b4")
            axes[0].plot(history.get("val_accs", []), label="Validation Accuracy", color="#ff7f0e")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Training vs Validation Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Loss curve
            axes[1].plot(history.get("train_losses", []), label="Training Loss", color="#d62728")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Cross Entropy Loss")
            axes[1].set_title("Training Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # F1 curve
            axes[2].plot(history.get("val_f1s", []), label="Validation Macro-F1", color="#2ca02c")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Macro-F1")
            axes[2].set_title("Validation Macro-F1")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.config.training_curves_plot, dpi=300)
            plt.close()
            logger.info(f"Saved training curves plot at: {self.config.training_curves_plot}")
        except Exception as e:
            logger.error("Failed to generate training curves plot")
            raise CustomException(e, sys)

    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        """Exports the model to ONNX format with dynamic node/edge axes and validates it."""
        try:
            logger.info("Exporting trained model to ONNX format...")
            model.eval()

            onnx_path = str(self.config.onnx_model_path)
            self.config.onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

            dummy_x = x.clone().detach()
            dummy_edge_index = edge_index.clone().detach()

            torch.onnx.export(
                model,
                (dummy_x, dummy_edge_index),
                onnx_path,
                export_params=True,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                input_names=["node_features", "edge_index"],
                output_names=["logits"],
                dynamic_axes={
                    "node_features": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "logits": {0: "num_nodes"},
                },
            )
            logger.info(f"ONNX model exported to: {onnx_path}")

            # Verify ONNX structure
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure verified successfully by onnx.checker.")

            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {
                "node_features": dummy_x.cpu().numpy(),
                "edge_index": dummy_edge_index.cpu().numpy(),
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            logger.info(
                f"ONNX Runtime inference verified. Output shape: {ort_outputs[0].shape}"
            )
        except Exception as e:
            logger.error("Failed to export or verify ONNX model")
            raise CustomException(e, sys)
