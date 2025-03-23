from typing import Any, Callable, Dict, Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided

from .core.cloud_trainer import CloudTrainer
from .core.local_simulator import launch_remote_rl_simulation
from .config.rllib import RLlibConfig
from .config.sagemaker import SageMakerConfig

class RemoteConfig():
    """
    RemoteConfig provides a high-level wrapper for Ray RLlib algorithm configurations and SageMaker deployment settings,
    enabling easy management and deployment of remote reinforcement learning training jobs on AWS SageMaker.

    This class streamlines the workflow between local RLlib algorithm tuning, environment simulation 
    connected to our web server, and remote training job launching. 
    Users can configure RLlib and SageMaker settings separately, run local simulations to validate configurations, 
    and subsequently deploy cloud training jobs.

    Args:
        config (Optional[AlgorithmConfig]): An initial RLlib AlgorithmConfig to set up the environment
            and training configuration.

    Methods:
        sagemaker(role_arn, output_path, instance_type, instance_count, max_run, region):
            Set or update SageMaker deployment configurations.

        simulate(env, num_env_runners, num_envs_per_env_runner, region):
            Run a local simulation to prepare and test remote training integration.

        train():
            Launch a SageMaker training job with the provided RLlib and SageMaker configurations.

        to_dict():
            Convert internal configurations to a dictionary format compatible with RLlib.

        set_config(**kwargs):
            Dynamically update internal RLlib and SageMaker configurations.

        register_env(name, env_creator):
            Register a custom environment with RLlib for remote training, corresponding to Ray's
            `ray.tune.register_env` method.
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
            num_envs_per_env_runner = max(self._rllib.num_envs_per_env_runner, 1)
            
        if num_env_runners is NotProvided:
            num_env_runners = max(self._rllib.num_env_runners, 1)

        if env is NotProvided:
            env = self._rllib.env or "Walker2d-v5"
            
        self._rllib.num_envs_per_env_runner = num_envs_per_env_runner
        self._rllib.num_env_runners = num_env_runners
        self._rllib.env = env
        entry_point = self._rllib.entry_point
        
        remote_training_key = launch_remote_rl_simulation(env, num_env_runners, num_envs_per_env_runner, entry_point, final_region)
        
        self.remote_training_key = remote_training_key
        return remote_training_key
    
    def train(self):
        """
        Launch a training job on SageMaker using the current RLlib configuration.
        
        Note: SageMaker parameters must be configured using `config_sagemaker()` prior to training.
        """
        config_dict = self.to_dict()
        sagemaker_dict = config_dict.get("sagemaker", {})
        rllib_dict = config_dict.get("rllib", {})

        results = self._trainer.train(sagemaker_dict, rllib_dict)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        rllib_dict = self._rllib.to_dict()
        sagemaker_dict = self._sagemaker.to_dict()
        return {"sagemaker": sagemaker_dict, "rllib": rllib_dict}
    
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if "sagemaker" in k:
                self._sagemaker.set_config(**v)
            elif "rllib" in k:
                self._rllib.set_config(**v)
            else:
                print(f"Warning: No attribute '{k}' in RemoteConfig")

    def register_env(self, name: str, env_creator: Callable):
        
        if not env_creator:
            print("Error: No environment creator provided.")    
            return None
        try:
            env_instance = env_creator({})
        except Exception as e:
            print(f"Error: {e}")
            env_instance = env_creator()
        
        entry_point = f"{env_instance.__class__.__module__}:{env_instance.__class__.__name__}"
        
        self._rllib.entry_point = entry_point
        self._rllib.env = name
        
        print(f"Environment registered: {name} ({entry_point})")