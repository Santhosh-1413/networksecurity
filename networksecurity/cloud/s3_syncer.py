import os
import sys
import boto3
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging


class S3Sync:
    """
    Handles syncing of local files/folders to AWS S3 bucket
    """
    
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")
    
    def sync_folder_to_s3(self, folder_path: str, bucket_name: str, bucket_folder_name: str) -> bool:
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, folder_path)
                    s3_key = os.path.join(bucket_folder_name, relative_path).replace("\\", "/")
                    
                    logging.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                    self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            
            logging.info(f"Successfully synced {folder_path} to s3://{bucket_name}/{bucket_folder_name}")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def sync_file_to_s3(self, file_path: str, bucket_name: str, s3_key: str) -> bool:
        try:
            logging.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
            self.s3_client.upload_file(file_path, bucket_name, s3_key)
            logging.info(f"Successfully uploaded {file_path}")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def download_file_from_s3(self, bucket_name: str, s3_key: str, local_file_path: str) -> bool:
        try:
            logging.info(f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}")
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.s3_client.download_file(bucket_name, s3_key, local_file_path)
            logging.info(f"Successfully downloaded to {local_file_path}")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def list_s3_files(self, bucket_name: str, prefix: str = "") -> list:
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            raise NetworkSecurityException(e, sys)