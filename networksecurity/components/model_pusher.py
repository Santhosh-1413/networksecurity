import os
import sys
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from networksecurity.entity.config_entity import ModelPusherConfig
from networksecurity.cloud.s3_syncer import S3Sync


class ModelPusher:
    """
    Pushes trained model artifacts to S3 for deployment
    """
    
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.s3_sync = S3Sync()
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Push trained model and preprocessor to S3
        
        Returns:
            ModelPusherArtifact with S3 paths
        """
        try:
            logging.info("Starting model pusher")
            
            # Get trained model file path
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            
            # Define S3 bucket details
            bucket_name = self.model_pusher_config.bucket_name
            s3_model_key = self.model_pusher_config.s3_model_key_path
            
            # Upload model to S3
            logging.info(f"Uploading model to S3: s3://{bucket_name}/{s3_model_key}")
            self.s3_sync.sync_file_to_s3(
                file_path=trained_model_path,
                bucket_name=bucket_name,
                s3_key=s3_model_key
            )
            
            # Also upload preprocessor separately
            preprocessor_path = self.model_pusher_config.preprocessor_path
            s3_preprocessor_key = self.model_pusher_config.s3_preprocessor_key_path
            
            if os.path.exists(preprocessor_path):
                logging.info(f"Uploading preprocessor to S3: s3://{bucket_name}/{s3_preprocessor_key}")
                self.s3_sync.sync_file_to_s3(
                    file_path=preprocessor_path,
                    bucket_name=bucket_name,
                    s3_key=s3_preprocessor_key
                )
            
            # Upload final model folder (model.pkl used for inference)
            final_model_dir = "final_model"
            if os.path.exists(final_model_dir):
                logging.info(f"Uploading final_model folder to S3")
                self.s3_sync.sync_folder_to_s3(
                    folder_path=final_model_dir,
                    bucket_name=bucket_name,
                    bucket_folder_name="final_model"
                )
            
            logging.info("Model pusher completed successfully")
            
            # Create artifact
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=bucket_name,
                s3_model_path=s3_model_key
            )
            
            return model_pusher_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)