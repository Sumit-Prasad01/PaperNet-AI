from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str


@dataclass(frozen=True)
class ModelArchitectureConfig:
    hidden_dim: int
    num_layers: int
    heads: int
    dropout: float
    attention_dropout: float
    classifier_dropout_1: float
    classifier_dropout_2: float
    jk_mode: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    seed: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    edge_dropout_p: float
    label_smoothing: float
    max_grad_norm: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    lr_scheduler_min_lr: float
    log_every_n_epochs: int
    device: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_file: Path
    classification_report_file: Path
    confusion_matrix_plot: Path
    training_curves_plot: Path
    onnx_model_path: Path
    onnx_opset_version: int
