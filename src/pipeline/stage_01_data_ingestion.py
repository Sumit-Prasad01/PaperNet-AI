import sys
from typing import Tuple

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from src.config import ConfigurationManager
from src.components import DataIngestion
from utils.custom_exception import CustomException
from utils.logger import logger

STAGE_NAME = "Stage 01: Data Ingestion"


class DataIngestionTrainingPipeline:
    """Pipeline stage for loading and preparing graph data."""

    def __init__(self):
        pass

    def run(self) -> Tuple[Planetoid, Data]:
        """Runs data ingestion stage."""
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            dataset, data = data_ingestion.load_data()
            return dataset, data
        except Exception as e:
            logger.error(f"Failed at {STAGE_NAME}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        dataset, data = pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
