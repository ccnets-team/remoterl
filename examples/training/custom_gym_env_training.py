# examples/training/custom_gym_env_training.py
"""
Example: End-to-End Remote RL Training with Custom Gym Environment and Specified AWS Region using RemoteRL and RLlib

This comprehensive example demonstrates the complete workflow for configuring, simulating, and
launching a remote reinforcement learning (RL) training job on AWS SageMaker using a custom Gym
environment (`SimpleCorridor`) from Ray RLlib, explicitly specifying the AWS region to ensure
consistency between the local simulation and remote training infrastructure.

Key concepts demonstrated:
- Configuring RLlib's PPO algorithm (`PPOConfig`).
- Utilizing RemoteRL's `RemoteConfig` to manage RLlib and SageMaker configurations seamlessly.
- Registering custom Gym environments using RemoteRL, equivalent to RLlib's built-in
  `ray.tune.register_env` method.
- Specifying the AWS region explicitly in both simulation and training for alignment.
- Dynamically setting SageMaker deployment parameters prior to launching training jobs.

Dependencies:
- ray[rllib]
- remoterl

Usage:
    python custom_gym_env_training.py
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
    remote_config.register_env("corridor-env", lambda config: SimpleCorridor(config))

    # Simulate the environment locally specifying the AWS region explicitly
    training_key = remote_config.simulate(env="corridor-env", region="us-east-1")
    print("Remote Training Key:", training_key)

    # Output the current RLlib configuration for verification
    # remote_config.to_dict() has {"rllib": {<rllib_config>}, "sagemaker": {<sagemaker_config>}}
    print("RLlib configs:", remote_config.to_dict().get("rllib"))

    role_arn = input("Enter your SageMaker role ARN (or press enter to use dummy): ") or "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    output_path = input("Enter your S3 output path (or press enter to use dummy): ") or "s3://remoterl"

    if not output_path.startswith("s3://"):
        output_path = f"s3://{output_path}"
        print("s3 Output path:", output_path)

    # Configure SageMaker parameters explicitly specifying the AWS region
    remote_config.sagemaker(
        role_arn=role_arn,
        output_path=output_path,
        region="us-east-1",
    )

    print("Final configs:", remote_config.to_dict())

    # Launch the remote training job on AWS SageMaker
    training_result = remote_config.train()
    print("Training job submitted. Result:", training_result)


if __name__ == "__main__":
    main()