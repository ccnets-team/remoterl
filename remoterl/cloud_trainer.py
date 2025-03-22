# remoterl/cloud_trainer.py
###############################################################################
# RemoteRL: the main class for training and running an RL environment in SageMaker
###############################################################################
import warnings
import logging

# Suppress specific pydantic warning about the "json" field.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "json" in "MonitoringDatasetFormat" shadows an attribute in parent "Base"',
    category=UserWarning,
    module="pydantic._internal._fields"
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

from sagemaker.estimator import Estimator
from .config.sagemaker import get_image_uri

class CloudTrainer:
    def __init__(self):
        pass
        
    @staticmethod
    def train(sagemaker_dict: dict, rllib_dict: dict):
        role_arn = sagemaker_dict.get("role_arn")
        output_path = sagemaker_dict.get("output_path")
        region = sagemaker_dict.get("region")
        instance_type = sagemaker_dict.get("instance_type", "ml.g5.4xlarge")
        instance_count = sagemaker_dict.get("instance_count", 1)
        max_run = sagemaker_dict.get("max_run", 3600)
        if not role_arn or not output_path or not region:
            raise ValueError("Invalid SageMaker configuration: Please provide a valid role_arn, output_path, and region.")
        
        image_uri = get_image_uri(region)

        estimator = Estimator(
            image_uri=image_uri,
            role=role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            output_path=output_path,
            max_run=max_run,
            region=region,
            hyperparameters=rllib_dict
        )
        estimator.fit()
        return estimator