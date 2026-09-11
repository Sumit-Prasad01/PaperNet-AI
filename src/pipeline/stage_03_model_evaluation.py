import sys
from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data

from src.config import ConfigurationManager
from src.components import ModelEvaluation
from src.models import AdvancedCoraGAT
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from utils.custom_exception import CustomException
from utils.helpers import get_device
from utils.logger import logger

STAGE_NAME = "Stage 03: Model Evaluation & Export"


class ModelEvaluationPipeline:
    """Pipeline stage for model evaluation, diagnostic visualization, and ONNX export."""

    def __init__(self):
        pass

    def run(
        self,
        model: Optional[torch.nn.Module] = None,
        data: Optional[Data] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs model evaluation and ONNX export."""
        try:
            config_manager = ConfigurationManager()
            eval_config = config_manager.get_model_evaluation_config()
            trainer_config = config_manager.get_model_trainer_config()
            arch_config = config_manager.get_model_architecture_config()

            if data is None:
                logger.info("Data object not provided; fetching via Stage 01...")
                stage1 = DataIngestionTrainingPipeline()
                dataset, data = stage1.run()

            device = get_device(trainer_config.device)

            if model is None:
                logger.info(
                    f"Model not provided in-memory; loading saved checkpoint from: {trainer_config.model_path}"
                )
                checkpoint = torch.load(trainer_config.model_path, map_location=device)
                model = AdvancedCoraGAT(
                    input_dim=checkpoint["input_dim"],
                    hidden_dim=checkpoint["hidden_dim"],
                    num_classes=checkpoint["num_classes"],
                    num_layers=checkpoint["num_layers"],
                    heads=checkpoint["heads"],
                    dropout=checkpoint.get("dropout", arch_config.dropout),
                    attention_dropout=checkpoint.get(
                        "attention_dropout", arch_config.attention_dropout
                    ),
                    classifier_dropout_1=arch_config.classifier_dropout_1,
                    classifier_dropout_2=arch_config.classifier_dropout_2,
                    jk_mode=arch_config.jk_mode,
                ).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])

            evaluator = ModelEvaluation(config=eval_config)
            metrics = evaluator.evaluate(model=model, data=data, history=history)
            return metrics
        except Exception as e:
            logger.error(f"Failed at {STAGE_NAME}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationPipeline()
        metrics = pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
