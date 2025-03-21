from typing import Optional, Dict, Any
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from remoterl.config.sagemaker import SageMakerConfig
from remoterl.cloud_trainer import CloudTrainer
from remoterl.utils.remote_utils import do_simulation  # Assumes you have a simulation utility
from ray.tune.registry import get_trainable_cls  # Assumes you have a function to get trainable classes

def extract_modified_config(selected_config, base_config):
    # Create a new dictionary with keys whose values differ or don't exist in the base_config.
    return {
        key: selected_config[key]
        for key in selected_config
        if key not in base_config or selected_config[key] != base_config[key]
    }

class RemoteConfig(AlgorithmConfig):
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
        
        self.trainable_name = None
        self.remote_training_key = None
        
        self.__default_config: Optional[AlgorithmConfig] = None
        
        # SageMaker configuration is not required initially.
        self._sagemaker: Optional[SageMakerConfig] = SageMakerConfig(region=None)
        
        # Cloud trainer for launching training jobs.
        self._trainer = CloudTrainer()
        
        self._internal_keys = set(self.__dict__.keys())
        
        self._init_config(config)
        
    def _build_default_config(self, trainable_name) -> AlgorithmConfig:
        return (
            get_trainable_cls(trainable_name)
            .get_default_config()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        )

    def _init_config(self, config: Optional[AlgorithmConfig] = None) -> "AlgorithmConfig":
        # Use the instance's trainable_name for default config.
        self.trainable_name = config.algo_class.__name__ if config is not None else "PPO"
        
        default_config = self._build_default_config(self.trainable_name)
        self.__default_config = default_config
        algorithm_config = default_config.copy()
        
        config_dict = config.to_dict()
        config_dict.pop("enable_rl_module_and_learner", None)
        config_dict.pop("enable_env_runner_and_connector_v2", None)
        
        algorithm_config = algorithm_config.from_dict(config_dict)
        super().__init__(self.trainable_name)
        super().update_from_dict(algorithm_config.to_dict())
            
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
        region = region or self._sagemaker.region or SageMakerConfig.BASE_REGION
        self._sagemaker.region = region
        
        self._sagemaker = SageMakerConfig(
            role_arn=role_arn,
            output_path=output_path,
            region=region,
            instance_type=instance_type,
            instance_count=instance_count,
            max_run=max_run
        )
    
    def simulate(
        self,
        env_type = "custom_gym",
        env: str = NotProvided,
        num_env_runners: int = NotProvided,
        num_envs_per_env_runner: int = NotProvided,
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
        region = region or self._sagemaker.region or SageMakerConfig.BASE_REGION
        self._sagemaker.region = region
        
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
        
        remote_training_key = do_simulation(
            env_type, self.env, self.num_envs_per_env_runner, self.num_env_runners, self._sagemaker.region  
        )
        
        self.remote_training_key = remote_training_key
        return remote_training_key
    
    def _remove_internal_keys(self, config_dict: dict):
        for key in self._internal_keys:
            config_dict.pop(key, None)
        config_dict.pop("_internal_keys", None)
        return config_dict
    
    def train(self):
        """
        Launch a training job on SageMaker using the current RLlib configuration.
        
        Note: SageMaker parameters must be configured using `config_sagemaker()` prior to training.
        """
        config_dict = self.to_dict()
        config_dict = self._remove_internal_keys(config_dict)
        
        config_dict["trainable_name"] = self.trainable_name
        config_dict["remote_training_key"] = self.remote_training_key
        
        results = self._trainer.train(self._sagemaker, config_dict)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        default_config = self.__default_config.to_dict()
        current_config = super(RemoteConfig, self).to_dict()
        current_config = self._remove_internal_keys(current_config)
        
        modified_config = extract_modified_config(current_config, default_config)
        modified_config["trainable_name"] = self.trainable_name
        modified_config["remote_training_key"] = self.remote_training_key

        sagemaker_dict = self._sagemaker.to_dict()
        # check out the duplicated configuration
        duplicate_keys = modified_config.keys() & sagemaker_dict.keys()
        if duplicate_keys:
            print("Warning: The following keys are duplicated in both configurations:", duplicate_keys)

        return {**modified_config, **sagemaker_dict}
  