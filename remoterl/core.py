# remote_rl/core.py
###############################################################################
# RemoteRL: the main class for training and running an RL environment in SageMaker
###############################################################################
import time
import boto3
from sagemaker import Model 
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor

from .config.sagemaker import SageMakerConfig
from .config.rllib import RLLibConfig
from .inference_api import InferenceAPI

class RemoteRL:

    def __init__(self):
        """
        Currently unused as RemoteRL only has static methods.
        """
        pass
        
    @staticmethod
    def train(sagemaker_config: SageMakerConfig, rllib_config: RLLibConfig):
        """
        Launch a SageMaker training job for your RemoteRL environment.

        This method packages up your environment, rllib_config, and the dynamically
        computed Docker image reference (using `get_image_uri("trainer")` from the 
        SageMakerConfig) into a SageMaker Estimator. Then it calls `estimator.fit()` 
        to run a cloud-based training job.

        **Usage Example**::

            from remote_rl import RemoteRL
            from src.config.aws_config import SageMakerConfig
            from src.config.rllib_config import RLLibConfig

            sagemaker_cfg = SageMakerConfig(...)
            rllib_config = RLLibConfig(...)

            # Kick off training in the cloud
            estimator = RemoteRL.train(sagemaker_cfg, rllib_config)
            print("Training job submitted:", estimator.latest_training_job.name)

        :param sagemaker_config: 
            A SageMakerConfig containing details like `role_arn`, `region`, and other 
            configuration settings. The container image URI is computed via 
            `sagemaker_config.get_image_uri("trainer")`.
        :param rllib_config:
            A RLLibConfig object with fields needed to configure environment, RL training, 
            and additional settings.
        :return:
            A `sagemaker.estimator.Estimator` instance that has started 
            the training job. You can query `.latest_training_job` for status.
        """
        trainer_config = sagemaker_config.trainer
        
        # Check for default output_path
        if trainer_config.output_path == trainer_config.DEFAULT_OUTPUT_PATH:
            raise ValueError("Invalid output_path: Please update the SageMaker trainer output_path to a valid S3 location.")
        
        image_uri = sagemaker_config.get_image_uri("trainer")
        rllib_config_dict = rllib_config.to_dict()

        estimator = Estimator(
            image_uri=image_uri,
            role=sagemaker_config.role_arn,
            instance_type=trainer_config.instance_type,
            instance_count=trainer_config.instance_count,
            output_path=trainer_config.output_path,
            max_run=trainer_config.max_run,
            region=sagemaker_config.region,
            rllib_config=rllib_config_dict
        )
        estimator.fit()
        return estimator
        
    @staticmethod
    def infer(sagemaker_config: SageMakerConfig):
        
        inference_config = sagemaker_config.inference
        # Check for default model_data
        if inference_config.model_data == inference_config.DEFAULT_MODEL_DATA:
            raise ValueError("Invalid model_data: Please update the SageMaker inference model_data to a valid S3 location.")
        
        image_uri = sagemaker_config.get_image_uri("inference")
        model = Model(
            role=sagemaker_config.role_arn,
            image_uri=image_uri,
            model_data=inference_config.model_data
        )
        print("Created SageMaker Model:", model)
        
        endpoint_name = inference_config.endpoint_name

        if not endpoint_name:
            endpoint_name = f"remoterl-{int(time.time())}"
            
        print("Using endpoint name:", endpoint_name)

        sagemaker_client = boto3.client("sagemaker", region = sagemaker_config.region)
        try:
            desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = desc["EndpointStatus"]
            print(f"Endpoint '{endpoint_name}' exists (status: {endpoint_status}).")
            endpoint_exists = True
        except sagemaker_client.exceptions.ClientError:
            endpoint_exists = False

        if endpoint_exists:
            print(f"Reusing existing endpoint: {endpoint_name}")
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=model.sagemaker_session
            )
        else:
            print(f"Creating a new endpoint: {endpoint_name}")
            new_predictor = model.deploy(
                initial_instance_count=inference_config.instance_count,
                instance_type=inference_config.instance_type,
                endpoint_name=endpoint_name
            )
            print("Deployed model to endpoint:", new_predictor)
            
            if new_predictor is not None:
                predictor = new_predictor
            else:
                print("model.deploy(...) returned None, creating Predictor manually...")
                predictor = Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=model.sagemaker_session
                )    

        # Return a RemoteRLAPI client for inference calls
        return InferenceAPI(predictor)