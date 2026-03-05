from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
import sys

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        raise NetworkSecurityException(e, sys)
