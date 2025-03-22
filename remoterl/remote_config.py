from typing import Optional, Dict, Any
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.tune.registry import get_trainable_cls  # Assumes you have a function to get trainable classes
from .config.sagemaker import SageMakerConfig
from .cloud_trainer import CloudTrainer
from .utils.simulation_utils import launch_remote_rl_simulation  # Assumes you have a simulation utility
from .config.rllib import RLlibConfig

class RemoteConfig():
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
        super().__init__()
 
        # SageMaker configuration is not required initially.
        self._sagemaker: Optional[SageMakerConfig] = SageMakerConfig()
        self._rllib: Optional[RLlibConfig] = RLlibConfig(config=config)
        
        # Cloud trainer for launching training jobs.
        self._trainer = CloudTrainer()
        
    def sagemaker(
        self,
        role_arn: str,
        output_path: str,
        instance_type: str = "ml.g5.4xlarge",
        instance_count: int = 1,
        max_run: int = 3600,
        region: str = None,
    ):
        """
        Configure or update the SageMaker deployment parameters.
        This should be called prior to launching a training job.
        """
        if region:
            final_region = region
        elif self._sagemaker.region and self._sagemaker.region != SageMakerConfig.DEFAULT_REGION:
            final_region = self._sagemaker.region
        else:
            final_region = SageMakerConfig.BASE_REGION              
        self._sagemaker.region = final_region
        
        self._sagemaker = SageMakerConfig(
            role_arn=role_arn,
            output_path=output_path,
            region=self._sagemaker.region,
            instance_type=instance_type,
            instance_count=instance_count,
            max_run=max_run
        )
    
    def simulate(
        self,
        env_type = "rllib",
        env: str = NotProvided,
        num_env_runners: int = NotProvided,
        num_envs_per_env_runner: int = NotProvided,
        entry_point = None,
        env_dir = None, 
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
        
        if region:
            final_region = region
        elif self._sagemaker.region and self._sagemaker.region != SageMakerConfig.DEFAULT_REGION:
            final_region = self._sagemaker.region
        else:
            final_region = SageMakerConfig.BASE_REGION        
        self._sagemaker.region = final_region
        
        # Use defaults if not already set in the algorithm configuration.
        if num_envs_per_env_runner is NotProvided:
            self.num_envs_per_env_runner = max(self.num_envs_per_env_runner, 1)
        else:
            self.num_envs_per_env_runner = num_envs_per_env_runner
            
        if num_env_runners is NotProvided:
            self.num_env_runners = max(self.num_env_runners, 1)
        else:
            self.num_env_runners = num_env_runners
        
        if env is NotProvided:
            self.env = self.env or "Walker2d-v5"
        else:
            self.env = env
        
        remote_training_key = launch_remote_rl_simulation(
            env_type, self.env, self.num_envs_per_env_runner, self.num_env_runners, entry_point, env_dir,
            self._sagemaker.region  
        )
        
        self.remote_training_key = remote_training_key
        return remote_training_key
    
    def train(self):
        """
        Launch a training job on SageMaker using the current RLlib configuration.
        
        Note: SageMaker parameters must be configured using `config_sagemaker()` prior to training.
        """
        config_dict = self.to_dict()
        rllib_dict = config_dict.get("rllib", {})
        sagemaker_dict = config_dict.get("sagemaker", {})

        results = self._trainer.train(sagemaker_dict, rllib_dict)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        rllib_dict = self._rllib.to_dict()
        sagemaker_dict = self._sagemaker.to_dict()
        return {"rllib": rllib_dict, "sagemaker": sagemaker_dict}
    
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if "sagemaker" in k:
                self._sagemaker.set_config(**v)
            elif "rllib" in k:
                self._rllib.set_config(**v)
            else:
                print(f"Warning: No attribute '{k}' in RemoteConfig")
