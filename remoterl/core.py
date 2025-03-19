# remote_rl/core.py
###############################################################################
# RemoteRL: the main class for training and running an RL environment in SageMaker
###############################################################################
from sagemaker.estimator import Estimator

from .config.sagemaker import SageMakerConfig
from .config.rllib import RemoteRLlibConfig

class RemoteRL:
    def __init__(self):
        pass
        
    @staticmethod
    def train(sagemaker_config: SageMakerConfig, hyperparameters):
        
        # Check for default output_path
        if sagemaker_config.output_path == sagemaker_config.DEFAULT_OUTPUT_PATH:
            raise ValueError("Invalid output_path: Please update the SageMaker output_path to a valid S3 location.")
        if sagemaker_config.region == sagemaker_config.DEFAULT_REGION:
            raise ValueError("Invalid region: Please update the SageMaker region to a valid AWS region.")
        
        image_uri = sagemaker_config.get_image_uri()

        estimator = Estimator(
            image_uri=image_uri,
            role=sagemaker_config.role_arn,
            instance_type=sagemaker_config.instance_type,
            instance_count=sagemaker_config.instance_count,
            output_path=sagemaker_config.output_path,
            max_run=sagemaker_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparameters
        )
        estimator.fit()
        return estimator