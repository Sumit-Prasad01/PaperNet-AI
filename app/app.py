"""PaperNet-AI: Research Topic Classification FastAPI Service.

Production-grade FastAPI inference API for classifying scientific papers into
research topics using Graph Attention Networks v2 (GATv2) with Jumping Knowledge.
Integrates with the PaperNet-AI modular architecture:
  - ConfigurationManager & config/config.yaml
  - PredictionPipeline (PyTorch & ONNX Runtime backends)
  - DataIngestion & artifacts/data/cora
  - Centralized logger & custom exception handling
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.constants import CONFIG_FILE_PATH, PROJECT_ROOT
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.pipeline import PredictionPipeline, CORA_CLASS_LABELS
from utils.custom_exception import CustomException
from utils.helpers import load_json
from utils.logger import logger

# --------------------------------------------------------------------------
# Constants & Directory Paths
# --------------------------------------------------------------------------
BASE_DIR: Path = PROJECT_ROOT
STATIC_DIR: Path = BASE_DIR / "static"
CONFIG_PATH: Path = CONFIG_FILE_PATH

NUM_FEATURES: int = 1433
NUM_CLASSES: int = 7

# --------------------------------------------------------------------------
# Application Setup & CORS
# --------------------------------------------------------------------------
app = FastAPI(
    title="PaperNet-AI: Research Topic Classification API",
    description=(
        "Production inference service for graph-based scientific paper topic "
        "classification using Graph Attention Networks (GATv2) and ONNX Runtime / PyTorch."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Shared State & Lazy Loading Helpers
# --------------------------------------------------------------------------
_predictor: Optional[PredictionPipeline] = None
_cora_graph_cache: Optional[Any] = None


def get_prediction_pipeline() -> PredictionPipeline:
    """Initializes or returns the singleton PredictionPipeline."""
    global _predictor
    if _predictor is None:
        try:
            logger.info("Initializing PredictionPipeline for FastAPI service...")
            _predictor = PredictionPipeline()
            logger.info("PredictionPipeline initialized successfully.")
        except Exception as exc:
            logger.error(f"Failed to initialize PredictionPipeline: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference model not ready or artifacts missing: {exc}. Run 'python main.py' to train first.",
            ) from exc
    return _predictor


def get_cora_graph_data() -> Any:
    """Lazily loads and caches the Cora dataset graph from artifacts/data."""
    global _cora_graph_cache
    if _cora_graph_cache is None:
        try:
            logger.info("Loading Cora citation graph via DataIngestion component...")
            config_mgr = ConfigurationManager(CONFIG_PATH)
            data_cfg = config_mgr.get_data_ingestion_config()
            ingestion = DataIngestion(data_cfg)
            _, data = ingestion.load_data()
            _cora_graph_cache = data
            logger.info(
                f"Cora graph loaded successfully: {data.num_nodes} nodes, {data.num_edges} edges."
            )
        except Exception as exc:
            logger.error(f"Failed to load Cora graph dataset: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cora dataset not available in {CONFIG_PATH}: {exc}",
            ) from exc
    return _cora_graph_cache


# --------------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------------
class GraphPredictRequest(BaseModel):
    """Payload for custom graph inference."""
    node_features: List[List[float]] = Field(
        ...,
        description=f"List of node feature vectors. Each vector must contain exactly {NUM_FEATURES} values.",
    )
    edge_index: Optional[List[List[int]]] = Field(
        None,
        description="[2, num_edges] COO edge connectivity array. If omitted, self-loops are used.",
    )
    edge_indices: Optional[List[List[int]]] = Field(
        None,
        description="Backwards-compatible alias for edge_index.",
    )
    backend: Optional[str] = Field(
        "onnx",
        description="Inference backend: 'onnx' (default, high-performance) or 'pytorch'.",
    )

    @field_validator("node_features")
    @classmethod
    def validate_features(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("node_features cannot be empty")
        for idx, row in enumerate(value):
            if len(row) != NUM_FEATURES:
                raise ValueError(
                    f"Node {idx} feature vector dimension must be {NUM_FEATURES}, got {len(row)}"
                )
        return value


class CoraNodeRequest(BaseModel):
    """Payload for predicting topic class on existing Cora benchmark nodes."""
    node_indices: List[int] = Field(
        ...,
        min_length=1,
        description="List of node indices to classify (0 to 2707).",
    )
    backend: Optional[str] = Field(
        "onnx",
        description="Inference backend: 'onnx' (default) or 'pytorch'.",
    )


class NodePrediction(BaseModel):
    """Classification prediction details for an individual node."""
    node_index: int
    predicted_class_id: int
    predicted_class_name: str
    topic_label: str
    confidence: float
    probabilities: List[float]
    logits: List[float]


class PredictionResponse(BaseModel):
    """API response model for graph classification inference."""
    num_nodes: int
    num_edges: int
    backend: str
    predictions: List[NodePrediction]


# --------------------------------------------------------------------------
# API Routes
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_home_page():
    """Serves the interactive web interface from the static directory."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "PaperNet-AI: Research Paper Topic Classification API",
        "status": "online",
        "docs": "/docs",
        "info": "/info",
        "health": "/health",
    }


@app.get("/health", summary="Health Check")
def health_check():
    """Returns server and inference runtime health status."""
    try:
        predictor = get_prediction_pipeline()
        onnx_session = predictor.load_onnx_model()
        providers = onnx_session.get_providers()
        device_name = str(predictor.device)
        status_str = "healthy"
    except Exception as exc:
        logger.warning(f"Health check warning: {exc}")
        return {
            "status": "degraded",
            "message": "Model artifacts not fully loaded. Run training pipeline first.",
            "error": str(exc),
        }

    return {
        "status": status_str,
        "service": "PaperNet-AI Inference Service",
        "device": device_name,
        "onnx_providers": providers,
        "cora_nodes_available": 2708,
    }


@app.get("/info", summary="Model & Architecture Metadata")
def model_info():
    """Returns comprehensive metadata about the trained model, ONNX inputs/outputs, and class mapping."""
    try:
        predictor = get_prediction_pipeline()
        onnx_info = predictor.get_onnx_info()
        inputs = onnx_info["inputs"]
        outputs = onnx_info["outputs"]
        providers = onnx_info["providers"]
    except Exception:
        inputs = [
            {"name": "node_features", "shape": ["num_nodes", NUM_FEATURES], "type": "tensor(float)"},
            {"name": "edge_index", "shape": [2, "num_edges"], "type": "tensor(int64)"},
        ]
        outputs = [{"name": "logits", "shape": ["num_nodes", NUM_CLASSES], "type": "tensor(float)"}]
        providers = ["CPUExecutionProvider"]

    return {
        "model_name": "PaperNet-AI Advanced GATv2",
        "Model Name": "PaperNet-AI Advanced GATv2",
        "architecture": "Graph Attention Network v2 with Jumping Knowledge (cat)",
        "feature_dimension": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "class_mapping": CORA_CLASS_LABELS,
        "providers": providers,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": {
            "name": "Cora Citation Network",
            "total_nodes": 2708,
            "total_edges": 10556,
            "classes": list(CORA_CLASS_LABELS.values()),
        },
    }


@app.get("/metrics", summary="Model Evaluation Metrics")
def evaluation_metrics():
    """Returns training evaluation metrics from artifacts/evaluation/metrics.json if present."""
    config_mgr = ConfigurationManager(CONFIG_PATH)
    eval_cfg = config_mgr.get_model_evaluation_config()
    metrics_path = eval_cfg.metrics_file

    if metrics_path.exists():
        try:
            return load_json(metrics_path)
        except Exception as exc:
            logger.error(f"Failed to read metrics file: {exc}")
            raise HTTPException(
                status_code=500, detail=f"Error reading metrics from {metrics_path}: {exc}"
            ) from exc

    return {
        "message": "Evaluation metrics not found. Run 'python main.py' to generate metrics.",
        "expected_path": str(metrics_path),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict on Custom Graph")
def predict_custom_graph(
    request: GraphPredictRequest,
    backend: Optional[str] = Query(None, description="Override backend ('onnx' or 'pytorch')"),
) -> PredictionResponse:
    """Classifies nodes for an arbitrary user-supplied graph structure and feature vectors."""
    try:
        predictor = get_prediction_pipeline()
        chosen_backend = (backend or request.backend or "onnx").lower()
        if chosen_backend not in ("onnx", "pytorch"):
            raise HTTPException(status_code=400, detail="Backend must be 'onnx' or 'pytorch'")

        features_np = np.array(request.node_features, dtype=np.float32)
        num_nodes = int(features_np.shape[0])

        # Resolve edge connectivity
        raw_edges = request.edge_index if request.edge_index is not None else request.edge_indices
        if raw_edges is not None:
            edge_arr = np.array(raw_edges, dtype=np.int64)
            if edge_arr.ndim != 2 or edge_arr.shape[0] != 2:
                raise HTTPException(
                    status_code=422,
                    detail="edge_index must have shape [2, num_edges]",
                )
            if edge_arr.size > 0:
                min_idx, max_idx = int(edge_arr.min()), int(edge_arr.max())
                if min_idx < 0 or max_idx >= num_nodes:
                    raise HTTPException(
                        status_code=422,
                        detail=f"edge_index node references out of range [0, {num_nodes - 1}]",
                    )
        else:
            # Self-loops default
            ids = np.arange(num_nodes, dtype=np.int64)
            edge_arr = np.vstack([ids, ids])

        num_edges = int(edge_arr.shape[1])
        target_indices = list(range(num_nodes))

        if chosen_backend == "pytorch":
            import torch
            x_tensor = torch.from_numpy(features_np)
            edge_tensor = torch.from_numpy(edge_arr)
            raw_preds = predictor.predict(
                x=x_tensor,
                edge_index=edge_tensor,
                node_indices=target_indices,
            )
        else:
            raw_preds = predictor.predict_onnx(
                x=features_np,
                edge_index=edge_arr,
                node_indices=target_indices,
            )

        predictions = [NodePrediction(**p) for p in raw_preds]

        return PredictionResponse(
            num_nodes=num_nodes,
            num_edges=num_edges,
            backend=chosen_backend,
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Inference failed during custom graph prediction: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Custom graph prediction failed: {exc}"
        ) from exc


@app.post(
    "/predict/cora_node",
    response_model=PredictionResponse,
    summary="Predict on Cora Dataset Benchmark Nodes",
)
def predict_cora_nodes(
    request: CoraNodeRequest,
    backend: Optional[str] = Query(None, description="Override backend ('onnx' or 'pytorch')"),
) -> PredictionResponse:
    """Classifies real research paper nodes from the official Cora citation network graph."""
    try:
        predictor = get_prediction_pipeline()
        data = get_cora_graph_data()
        chosen_backend = (backend or request.backend or "onnx").lower()
        if chosen_backend not in ("onnx", "pytorch"):
            raise HTTPException(status_code=400, detail="Backend must be 'onnx' or 'pytorch'")

        max_valid_idx = int(data.num_nodes) - 1
        invalid = [i for i in request.node_indices if i < 0 or i > max_valid_idx]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Node indices out of bounds {invalid}. Cora graph contains nodes 0 to {max_valid_idx}.",
            )

        if chosen_backend == "pytorch":
            raw_preds = predictor.predict(
                x=data.x,
                edge_index=data.edge_index,
                node_indices=request.node_indices,
            )
        else:
            x_np = data.x.cpu().numpy()
            edge_np = data.edge_index.cpu().numpy()
            raw_preds = predictor.predict_onnx(
                x=x_np,
                edge_index=edge_np,
                node_indices=request.node_indices,
            )

        predictions = [NodePrediction(**p) for p in raw_preds]

        return PredictionResponse(
            num_nodes=int(data.num_nodes),
            num_edges=int(data.edge_index.shape[1]),
            backend=chosen_backend,
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Inference failed on Cora node prediction: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Cora node prediction failed: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Mount Static Files (Mount after explicit routes so API routes take precedence)
# --------------------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static_dir")
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static_root")
else:
    logger.warning(f"Static directory not found at: {STATIC_DIR}")


# --------------------------------------------------------------------------
# Direct execution support
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting PaperNet-AI FastAPI application server...")
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)