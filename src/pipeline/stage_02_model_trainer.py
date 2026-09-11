import sys
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from src.config import ConfigurationManager
from src.components import ModelTrainer
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from utils.custom_exception import CustomException
from utils.logger import logger

STAGE_NAME = "Stage 02: Model Trainer"


class ModelTrainerPipeline:
    """Pipeline stage for training the Advanced GATv2 model."""

    def __init__(self):
        pass

    def run(
        self,
        dataset: Optional[Planetoid] = None,
        data: Optional[Data] = None,
    ) -> Tuple[torch.nn.Module, Dict[str, Any], Data]:
        """Runs model training stage."""
        try:
            if dataset is None or data is None:
                logger.info("Dataset/Data not provided; fetching via Stage 01...")
                stage1 = DataIngestionTrainingPipeline()
                dataset, data = stage1.run()

            config_manager = ConfigurationManager()
            trainer_config = config_manager.get_model_trainer_config()
            arch_config = config_manager.get_model_architecture_config()

            model_trainer = ModelTrainer(
                trainer_config=trainer_config,
                arch_config=arch_config,
                dataset=dataset,
                data=data,
            )
            model, history = model_trainer.train()
            return model, history, data
        except Exception as e:
            logger.error(f"Failed at {STAGE_NAME}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainerPipeline()
        model, history, data = pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
