from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

@dataclass
class Exploration:
    """Defines exploration parameters compatible with Ray RLLib exploration configs."""
    type: str = "GaussianNoise"  # "GaussianNoise", "EpsilonGreedy", "OrnsteinUhlenbeck", etc.
    initial_scale: Optional[float] = None
    final_scale: Optional[float] = None
    scale_timesteps: Optional[int] = None
    epsilon_timesteps: Optional[int] = None
    initial_epsilon: Optional[float] = None
    final_epsilon: Optional[float] = None
    ou_base_scale: Optional[float] = None
    ou_theta: Optional[float] = None
    ou_sigma: Optional[float] = None

    def to_dict(self):
        exploration_config = {"type": self.type}
        if self.type == "GaussianNoise":
            exploration_config.update({
                "initial_scale": self.initial_scale or 0.1,
                "final_scale": self.final_scale or 0.01,
                "scale_timesteps": self.scale_timesteps or 10000,
            })
        elif self.type == "EpsilonGreedy":
            exploration_config.update({
                "initial_epsilon": self.initial_epsilon or 1.0,
                "final_epsilon": self.final_epsilon or 0.01,
                "epsilon_timesteps": self.epsilon_timesteps or 10000,
            })
        elif self.type == "OrnsteinUhlenbeckNoise":
            exploration_config.update({
                "ou_base_scale": self.ou_base_scale or 0.1,
                "ou_theta": self.ou_theta or 0.15,
                "ou_sigma": self.final_scale or 0.2,
            })
        return exploration_config

@dataclass
class RLLibConfig:
    """RLLibConfig for Remote RL training using Ray RLLib and AWS SageMaker."""

    # Environment and Remote Gateway parameters
    env: str = "CartPole-v1"
    training_key: Optional[str] = None
    entry_point: Optional[str] = None
    env_dir: Optional[str] = None

    # Training control
    run_or_experiment: str = "PPO"
    training_iteration: int = 50
    checkpoint_at_end: bool = True

    # Parallelization and Scaling
    num_workers: int = 1
    num_envs_per_worker: int = 1

    # Core RL Algorithm Parameters
    gamma: float = 0.99
    lambda_: float = 0.95
    train_batch_size: int = 4000
    rollout_fragment_length: int = 200
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    clip_param: float = 0.2

    # Learning Rate
    lr: float = 1e-4
    lr_schedule: Optional[List[List[float]]] = None

    # Model configuration
    model: Dict[str, Any] = field(default_factory=lambda: {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu"
    })

    # Exploration
    exploration_config: Optional[Dict[str, Any]] = None

    # Custom environment configuration
    env_config: Dict[str, Any] = field(default_factory=dict)

    def set_exploration(self, exploration: Exploration):
        """Sets the exploration configuration in a way compatible with Ray RLLib."""
        self.model["exploration_config"] = exploration.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary ready for RLLib."""
        config_dict = asdict(self)
        if "exploration_config" in self.model:
            config_dict["exploration_config"] = self.model.pop("exploration_config")
        return config_dict
