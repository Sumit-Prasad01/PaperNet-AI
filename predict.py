import sys
import numpy as np
import torch
from src.pipeline import DataIngestionTrainingPipeline, PredictionPipeline
from utils.custom_exception import CustomException
from utils.logger import logger


def run_inference_demo():
    """Demonstrates inference using both PyTorch and ONNX models on test nodes."""
    try:
        logger.info("Initializing inference demonstration...")

        # Load graph data
        stage1 = DataIngestionTrainingPipeline()
        dataset, data = stage1.run()

        predictor = PredictionPipeline()

        # Select first 5 test nodes to classify
        test_indices = torch.where(data.test_mask)[0][:5].tolist()
        logger.info(f"Classifying sample test nodes: {test_indices}")

        # 1. PyTorch inference
        logger.info("Running PyTorch model inference...")
        pt_results = predictor.predict(
            x=data.x,
            edge_index=data.edge_index,
            node_indices=test_indices,
        )
        print("\n--- PyTorch Predictions ---")
        for res in pt_results:
            true_label_id = int(data.y[res["node_index"]].item())
            print(
                f"Node {res['node_index']} -> Predicted: {res['topic_label']} "
                f"(Class {res['predicted_class_id']}) with {res['confidence']*100:.2f}% confidence | "
                f"Ground Truth Class: {true_label_id}"
            )

        # 2. ONNX inference
        logger.info("Running ONNX model inference...")
        onnx_results = predictor.predict_onnx(
            x=data.x.cpu().numpy(),
            edge_index=data.edge_index.cpu().numpy(),
            node_indices=test_indices,
        )
        print("\n--- ONNX Runtime Predictions ---")
        for res in onnx_results:
            true_label_id = int(data.y[res["node_index"]].item())
            print(
                f"Node {res['node_index']} -> Predicted: {res['topic_label']} "
                f"(Class {res['predicted_class_id']}) with {res['confidence']*100:.2f}% confidence | "
                f"Ground Truth Class: {true_label_id}"
            )

    except CustomException as ce:
        logger.error(f"Custom Exception in inference demo:\n{ce}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled Exception in inference demo:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    run_inference_demo()
