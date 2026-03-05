from dataclasses import dataclass
import os
from networksecurity.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, pipeline_name)
    timestamp: str = PIPELINE_NAME

@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = None
    
    def __post_init__(self):
        self.data_ingestion_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = DATA_INGESTION_DATABASE_NAME

@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.data_validation_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.data_transformation_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCESSING_OBJECT_FILE_NAME
        )

@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.model_trainer_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD