from dataclasses import dataclass
from typing import Optional, Type, Dict, Any
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
    algorithm_config: AlgorithmConfig = None
    __default_config: AlgorithmConfig = None
    trainable_name: str = "PPO"
    remote_training_key: Optional[str] = None
    
    def __post_init__(self):
        # Initialize the algorithm configuration using from_config.
        self.__default_config = self._build_default_config(self.trainable_name)
        if self.algorithm_config is None:
            self.algorithm_config = self.__default_config.copy()
            # To show the default values in the UI, we need to set some values explicitly.
            self.algorithm_config = (
                self.algorithm_config
                .env_runners(rollout_fragment_length='auto', sample_timeout_s=60)
                .training(train_batch_size=1024, num_epochs=15, lr=1e-4)
            )
    
    @classmethod
    def _build_default_config(cls, trainable_name) -> AlgorithmConfig:
        return (
            get_trainable_cls(trainable_name)
            .get_default_config()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        )

    @classmethod
    def from_config(cls: Type["RemoteRLlibConfig"], config: Optional[AlgorithmConfig] = None) -> "RemoteRLlibConfig":
        instance = cls()
        if config is None:
            # Use the instance's trainable_name for default config.
            default_config = cls._build_default_config(instance.trainable_name)
            instance.__default_config = default_config
            instance.algorithm_config = default_config.copy()
        else:
            # Use the provided config's trainable name.
            print(f"Using trainable name: {instance.trainable_name}")
            trainable_name = config.algo_class.__name__
            default_config = cls._build_default_config(trainable_name)
            instance.__default_config = default_config
            instance.algorithm_config = default_config.from_dict(config.to_dict())
        return instance

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in RemoteRLlibConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        default_config = self.__default_config.to_dict()
        current_config = self.algorithm_config.to_dict()
        modified_config = extract_modified_config(current_config, default_config)
        modified_config["trainable_name"] = self.trainable_name
        modified_config["remote_training_key"] = self.remote_training_key
        return modified_config
