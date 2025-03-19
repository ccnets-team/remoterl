from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.registry import get_trainable_cls

def extract_modified_config(selected_config, base_config):
    # Create a new dictionary with keys whose values differ or don't exist in the base_config.
    return {
        key: selected_config[key]
        for key in selected_config
        if key not in base_config or selected_config[key] != base_config[key]
    }

@dataclass
class RemoteRLConfig:
    enable_rl_module_and_learner: bool = False
    enable_env_runner_and_connector_v2: bool = False
    enable_env_checking: bool = True
    trainable_name: str = "PPO"
    remote_training_key: Optional[str] = None
    def to_dict(self):
        return asdict(self)

@dataclass
class RLLibConfig:
    algorithm_config: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    remoterl_config: RemoteRLConfig = field(default_factory=RemoteRLConfig)
    
    def __post_init__(self):
        # Initialize the algorithm configuration with the specified parameters
        self.algorithm_config = self.create_defualt_config()
        self.algorithm_config = (
            self.algorithm_config
            .env_runners(rollout_fragment_length='auto', sample_timeout_s=60)
            .training(train_batch_size=1024, num_sgd_iter=15, lr=1e-4)
        )
    
    def create_defualt_config(self):
        default_config = (get_trainable_cls(self.remoterl_config.trainable_name)
                          .get_default_config()
                          .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                          .environment(disable_env_checking=True)
                          )
        
        return default_config

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in RLLibConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        # Obtain the default configuration for the specified algorithm
        default_config = self.create_defualt_config().to_dict()
        # Obtain the current configuration
        current_config = self.algorithm_config.to_dict()
        # Identify configurations that differ from the default
        modified_config = extract_modified_config(current_config, default_config)
        # Include remote RL configurations
        remoterl_dict = self.remoterl_config.to_dict()    
        # Merge modified configurations with remote RL settings
        hyperparameters_dict = {**modified_config, **remoterl_dict}
        return hyperparameters_dict
