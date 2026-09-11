import sys
import time

from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_model_trainer import ModelTrainerPipeline
from src.pipeline.stage_03_model_evaluation import ModelEvaluationPipeline
from utils.custom_exception import CustomException
from utils.logger import logger


class TrainingPipeline:
    """Orchestrates the complete end-to-end training, evaluation, and export workflow."""

    def __init__(self):
        pass

    def run_pipeline(self) -> None:
        """Executes all pipeline stages sequentially."""
        try:
            start_time = time.time()
            logger.info("==================================================")
            logger.info("       PaperNet-AI Training Pipeline Started      ")
            logger.info("==================================================")

            # Stage 1: Data Ingestion
            logger.info(">>>>>> Stage 01: Data Ingestion started <<<<<<")
            stage1 = DataIngestionTrainingPipeline()
            dataset, data = stage1.run()
            logger.info(">>>>>> Stage 01: Data Ingestion completed <<<<<<\n")

            # Stage 2: Model Trainer
            logger.info(">>>>>> Stage 02: Model Trainer started <<<<<<")
            stage2 = ModelTrainerPipeline()
            model, history, data = stage2.run(dataset=dataset, data=data)
            logger.info(">>>>>> Stage 02: Model Trainer completed <<<<<<\n")

            # Stage 3: Model Evaluation & ONNX Export
            logger.info(">>>>>> Stage 03: Model Evaluation started <<<<<<")
            stage3 = ModelEvaluationPipeline()
            metrics = stage3.run(model=model, data=data, history=history)
            logger.info(">>>>>> Stage 03: Model Evaluation completed <<<<<<\n")

            elapsed_time = time.time() - start_time
            logger.info("==================================================")
            logger.info(
                f"PaperNet-AI Pipeline Completed Successfully in {elapsed_time:.2f}s!"
            )
            logger.info(f"Final Test Accuracy: {metrics.get('test_accuracy', 0.0):.4f}")
            logger.info(f"Final Test Macro-F1: {metrics.get('test_macro_f1', 0.0):.4f}")
            logger.info("==================================================")

        except Exception as e:
            logger.error("Error occurred during pipeline execution")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.exception(e)
        raise e
