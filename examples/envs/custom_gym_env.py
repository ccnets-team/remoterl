# examples/envs/custom_gym_env.py
"""
Example: Remote RL Training with Custom Gym Environment using RemoteRL and RLlib

This example illustrates setting up and running remote reinforcement learning
training jobs using a custom Gym environment (`SimpleCorridor`) from Ray RLlib,
showing clearly how RLlib's built-in registration methods (`ray.tune.register_env`) 
correspond directly to RemoteRL's `register_env` method.

Key concepts demonstrated:
- RLlib's PPO algorithm configuration (`PPOConfig`).
- Correspondence between RLlib configurations and RemoteRL's `RemoteConfig`.
- Registering custom Gym environments.
- Simulating locally before launching a cloud training job.

Dependencies:
- ray[rllib]
- remoterl

Usage:
    python custom_gym_env.py
"""
from ray.rllib.algorithms.ppo import PPOConfig
from remoterl.remote_config import RemoteConfig
from ray.rllib.examples.envs.custom_gym_env import SimpleCorridor


def main():
    # Define RLlib PPO algorithm configuration
    config = (
        PPOConfig()
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=16,
            rollout_fragment_length="auto",
            sample_timeout_s=60,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .training(
            train_batch_size=64,
            num_epochs=10,
            lr=1e-4,
        )
    )

    # Initialize RemoteConfig with the RLlib algorithm configuration
    remote_config = RemoteConfig(config=config)

    # Register custom Gym environment with RemoteRL
    # Equivalent to RLlib's: tune.register_env("corridor-env", lambda config: SimpleCorridor(config))
    remote_config.register_env("corridor-env", lambda config: SimpleCorridor(config))

    # Simulate the environment locally and generate a remote training key
    training_key = remote_config.simulate(env="corridor-env")
    print("Remote Training Key:", training_key)

    # Output the current RLlib configuration for confirmation
    # remote_config.to_dict() has {"rllib": {<rllib_config>}, "sagemaker": {<sagemaker_config>}}
    print("RLlib configs:", remote_config.to_dict().get("rllib"))
    


if __name__ == "__main__":
    main()