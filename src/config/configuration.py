import sys
from pathlib import Path

from src.constants import CONFIG_FILE_PATH, PROJECT_ROOT
from src.entity import (
    DataIngestionConfig,
    ModelArchitectureConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from utils.custom_exception import CustomException
from utils.helpers import create_directories, read_yaml
from utils.logger import logger


class ConfigurationManager:
    """Manages reading configuration YAML files and providing structured config entities."""

    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        try:
            self.config = read_yaml(config_filepath)
            artifacts_root_str = self.config.get("artifacts_root", "artifacts")
            artifacts_root = (
                PROJECT_ROOT / artifacts_root_str
                if not Path(artifacts_root_str).is_absolute()
                else Path(artifacts_root_str)
            )
            create_directories([artifacts_root])
            logger.info(f"Initialized ConfigurationManager with config: {config_filepath}")
        except Exception as e:
            logger.error("Failed to initialize ConfigurationManager")
            raise CustomException(e, sys)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Constructs and returns DataIngestionConfig."""
        try:
            config = self.config["data_ingestion"]
            raw_root = Path(config["root_dir"])
            root_dir = PROJECT_ROOT / raw_root if not raw_root.is_absolute() else raw_root
            create_directories([root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=root_dir,
                dataset_name=config["dataset_name"],
            )
            return data_ingestion_config
        except Exception as e:
            logger.error("Failed to retrieve DataIngestionConfig")
            raise CustomException(e, sys)

    def get_model_architecture_config(self) -> ModelArchitectureConfig:
        """Constructs and returns ModelArchitectureConfig."""
        try:
            config = self.config["model_architecture"]
            model_architecture_config = ModelArchitectureConfig(
                hidden_dim=int(config["hidden_dim"]),
                num_layers=int(config["num_layers"]),
                heads=int(config["heads"]),
                dropout=float(config["dropout"]),
                attention_dropout=float(config["attention_dropout"]),
                classifier_dropout_1=float(config.get("classifier_dropout_1", 0.40)),
                classifier_dropout_2=float(config.get("classifier_dropout_2", 0.25)),
                jk_mode=str(config.get("jk_mode", "cat")),
            )
            return model_architecture_config
        except Exception as e:
            logger.error("Failed to retrieve ModelArchitectureConfig")
            raise CustomException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Constructs and returns ModelTrainerConfig."""
        try:
            config = self.config["model_trainer"]
            raw_root = Path(config["root_dir"])
            root_dir = PROJECT_ROOT / raw_root if not raw_root.is_absolute() else raw_root
            create_directories([root_dir])

            model_name = config["model_name"]
            model_path = root_dir / model_name

            model_trainer_config = ModelTrainerConfig(
                root_dir=root_dir,
                model_name=model_name,
                model_path=model_path,
                seed=int(config.get("seed", 42)),
                epochs=int(config.get("epochs", 400)),
                patience=int(config.get("patience", 60)),
                learning_rate=float(config.get("learning_rate", 0.003)),
                weight_decay=float(config.get("weight_decay", 0.0005)),
                edge_dropout_p=float(config.get("edge_dropout_p", 0.10)),
                label_smoothing=float(config.get("label_smoothing", 0.05)),
                max_grad_norm=float(config.get("max_grad_norm", 2.0)),
                lr_scheduler_factor=float(config.get("lr_scheduler_factor", 0.5)),
                lr_scheduler_patience=int(config.get("lr_scheduler_patience", 15)),
                lr_scheduler_min_lr=float(config.get("lr_scheduler_min_lr", 1e-5)),
                log_every_n_epochs=int(config.get("log_every_n_epochs", 10)),
                device=str(config.get("device", "auto")),
            )
            return model_trainer_config
        except Exception as e:
            logger.error("Failed to retrieve ModelTrainerConfig")
            raise CustomException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Constructs and returns ModelEvaluationConfig."""
        try:
            config = self.config["model_evaluation"]
            raw_root = Path(config["root_dir"])
            root_dir = PROJECT_ROOT / raw_root if not raw_root.is_absolute() else raw_root
            create_directories([root_dir])

            raw_onnx = Path(config["onnx_model_path"])
            onnx_model_path = PROJECT_ROOT / raw_onnx if not raw_onnx.is_absolute() else raw_onnx

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=root_dir,
                metrics_file=root_dir / config["metrics_file"],
                classification_report_file=root_dir / config["classification_report_file"],
                confusion_matrix_plot=root_dir / config["confusion_matrix_plot"],
                training_curves_plot=root_dir / config["training_curves_plot"],
                onnx_model_path=onnx_model_path,
                onnx_opset_version=int(config.get("onnx_opset_version", 18)),
            )
            return model_evaluation_config
        except Exception as e:
            logger.error("Failed to retrieve ModelEvaluationConfig")
            raise CustomException(e, sys)

