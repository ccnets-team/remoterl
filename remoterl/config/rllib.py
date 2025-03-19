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
class RemoteRLlibConfig:
    algorithm_config: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    default_config: AlgorithmConfig = field(init=False)
    trainable_name: str = "PPO"
    remote_training_key: Optional[str] = None
    
    def __post_init__(self):
        # Initialize the algorithm configuration using from_config.
        self.algorithm_config = self.from_config(self.algorithm_config)
        # Apply additional default settings (modify algorithm_config directly).
        self.algorithm_config = (
            self.algorithm_config
            .env_runners(rollout_fragment_length='auto', sample_timeout_s=60)
            .training(train_batch_size=1024, num_epochs=15, lr=1e-4)
        )
   
    @classmethod
    def from_config(cls, config: Optional[AlgorithmConfig] = None) -> AlgorithmConfig:
        # If no config is provided, use the default configuration for the trainable.
        if config is None:
            default_config = (
                get_trainable_cls(cls.trainable_name)  # using class variable
                .get_default_config()
                .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                .environment(disable_env_checking=True)
            )
            cls.default_config = default_config
            return default_config
        else:
            # If a config is provided, update the default config with the provided values.
            default_config = (
                get_trainable_cls(cls.trainable_name)
                .get_default_config()
                .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                .environment(disable_env_checking=True)
            )
            cls.default_config = default_config
            # Merge the provided config into the default config.
            merged_config = default_config.from_dict(config.to_dict())
            return merged_config

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in RemoteRLlibConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        default_config = self.default_config.to_dict()
        current_config = self.algorithm_config.to_dict()
        modified_config = extract_modified_config(current_config, default_config)
        modified_config["trainable_name"] = self.trainable_name
        modified_config["remote_training_key"] = self.remote_training_key
        return modified_config

    def simulate(self, **kwargs):
        raise NotImplementedError("Simulation is not yet implemented")