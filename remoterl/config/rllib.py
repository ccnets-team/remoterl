from typing import Optional, Dict, Any
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.tune.registry import get_trainable_cls  # Assumes you have a function to get trainable classes

def extract_modified_config(selected_config, base_config):
    # Create a new dictionary with keys whose values differ or don't exist in the base_config.
    return {
        key: selected_config[key]
        for key in selected_config
        if key not in base_config or selected_config[key] != base_config[key]
    }

class RLlibConfig(AlgorithmConfig):
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        # Accept an RLlib config at initialization or use a default.
        
        self.trainable_name = None
               
        self.remote_training_key = None
        self.env_type = None 
        self.env_id = None
        self.entry_point = None
        self.env_dir = None 
        
        self.__default_config: Optional[AlgorithmConfig] = None
        
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
        self.__default_config = default_config.copy()
        algorithm_config = (
            default_config
            .env_runners(rollout_fragment_length='auto', sample_timeout_s=120)
            .training(train_batch_size=4000, num_epochs=15, minibatch_size = 128, 
                      lr=1e-4)
        )
        if config:
            config_dict = config.to_dict()
            config_dict.pop("enable_rl_module_and_learner", None)
            config_dict.pop("enable_env_runner_and_connector_v2", None)
            algorithm_config = algorithm_config.from_dict(config_dict)
            
        super().__init__(self.trainable_name)
        super().update_from_dict(algorithm_config.to_dict())
            
    def _remove_internal_keys(self, config_dict: dict):
        for key in self._internal_keys:
            config_dict.pop(key, None)
        config_dict.pop("_internal_keys", None)
        return config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLlib."""
        default_config = self.__default_config.to_dict()
        current_config = super().to_dict()
        current_config = self._remove_internal_keys(current_config)
        rllib_config = extract_modified_config(current_config, default_config)
        rllib_config["trainable_name"] = self.trainable_name
        rllib_config["remote_training_key"] = self.remote_training_key

        return rllib_config
    
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in RemoteRLlibConfig")
