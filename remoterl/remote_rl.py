from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from remoterl.config.rllib import RemoteRLlibConfig
from remoterl.config.sagemaker import SageMakerConfig
from remoterl.cloud_trainer import CloudTrainer
from remoterl.utils.remote_utils import do_simulation  # Assumes you have a simulation utility

class RemoteRL:
    """
    RemoteRL provides a user-friendly interface to bridge local RLlib configurations
    with cloud-based training on SageMaker. It separates the concerns of algorithm 
    configuration and cloud deployment settings, enabling a smooth workflow:

      1. Initialize with an RLlib configuration or update it later.
      2. Run a simulation to obtain a remote training key for cloud integration.
      3. Configure SageMaker deployment parameters when ready to train.
      4. Launch training on SageMaker using the consolidated configuration.

    This design allows you to defer SageMaker-related settings until training time, 
    keeping local development lightweight and focused on RL algorithm tuning.
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        # Accept an RLlib config at initialization or use a default.
        self.rllib_config = RemoteRLlibConfig.from_config(config)
            
        # SageMaker configuration is not required initially.
        self.sagemaker_config: Optional[SageMakerConfig] = None
        
        # Cloud trainer for launching training jobs.
        self.cloud_trainer = CloudTrainer()
    
    def config_sagemaker(
        self,
        role_arn: str,
        output_path: str,
        region: str,
        instance_type: str = "ml.g5.4xlarge",
        instance_count: int = 1,
        max_run: int = 3600
    ):
        """
        Configure or update the SageMaker deployment parameters.
        This should be called prior to launching a training job.
        """
        self.sagemaker_config = SageMakerConfig(
            role_arn=role_arn,
            output_path=output_path,
            region=region,
            instance_type=instance_type,
            instance_count=instance_count,
            max_run=max_run
        )
    
    def update_rllib(self, rllib_config: AlgorithmConfig):
        """
        Replace the current RLlib configuration with a new one.
        This is useful when you need to update the algorithm settings during development.
        """
        self.rllib_config = RemoteRLlibConfig.from_config(rllib_config)
    
    def simulate(
        self,
        env_type: str = 'gym',
        region: Optional[str] = None
    ) -> str:
        """
        Run a simulation to bridge local environments with the cloud training infrastructure.
        
        The simulation sets up environment parameters from the current RLlib configuration 
        and generates a remote training key.
        
        Parameters:
          - env_type: The type of environment (e.g., 'gym' or 'unity').
          - region: (Optional) AWS region; if not provided, defaults to the SageMaker region or 'us-east-1'.
        
        Returns:
          A remote training key used for linking to the cloud training job.
        """
        # Determine the region: priority is given to the provided region,
        # then the SageMaker configuration, then a default value.
        given_region = region or (self.sagemaker_config.region if self.sagemaker_config else "us-east-1")
        if self.sagemaker_config:
            self.sagemaker_config.region = given_region
        
        algorithm_config = self.rllib_config.algorithm_config
        
        # Use defaults if not already set in the algorithm configuration.
        num_envs_per_env_runner = algorithm_config.num_envs_per_env_runner or 64
        self.rllib_config.algorithm_config.num_envs_per_env_runner = num_envs_per_env_runner
        
        num_env_runners = algorithm_config.num_env_runners or 4
        self.rllib_config.algorithm_config.num_env_runners = num_env_runners
        
        env = algorithm_config.env or "Walker2d-v5"
        self.rllib_config.algorithm_config.env = env
        
        remote_training_key = do_simulation(
            env_type, env, num_envs_per_env_runner, num_env_runners, given_region
        )
        self.rllib_config.remote_training_key = remote_training_key
        return remote_training_key
    
    def train(self):
        """
        Launch a training job on SageMaker using the current RLlib configuration.
        
        Note: SageMaker parameters must be configured using `config_sagemaker()` prior to training.
        """
        hyperparameters = self.rllib_config.to_dict()
        results = self.cloud_trainer.train(self.sagemaker_config, hyperparameters)
        return results
    
    def to_dict(self):
        """
        Return a dictionary representation of the RemoteRL object.
        """
        rllib_dict = self.rllib_config.to_dict()
        sagemaker_dict = self.sagemaker_config.to_dict() if self.sagemaker_config else {}
        # check out the duplicated configuration
        duplicate_keys = rllib_dict.keys() & sagemaker_dict.keys()
        if duplicate_keys:
            print("Warning: The following keys are duplicated in both configurations:", duplicate_keys)

        return {**self.rllib_config.to_dict(), **self.sagemaker_config.to_dict()}