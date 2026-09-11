import sys
from src.pipeline import TrainingPipeline
from utils.custom_exception import CustomException
from utils.logger import logger


def main():
    """Main application entrypoint for executing PaperNet-AI training pipeline."""
    try:
        logger.info("Starting PaperNet-AI End-to-End Pipeline Execution...")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logger.info("PaperNet-AI Pipeline Execution Finished Successfully.")
    except CustomException as ce:
        logger.error(f"Custom Exception caught in main:\n{ce}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled Exception in main:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
