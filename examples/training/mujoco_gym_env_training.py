# examples/training/mujoco_gym_env_training.py
"""
Example: End-to-End Remote RL Training with MuJoCo Environment and Specified AWS Region using RemoteRL and RLlib

This example demonstrates the complete process for configuring, simulating, and
launching a remote reinforcement learning (RL) training job on AWS SageMaker
using a standard MuJoCo environment ("Walker2d-v5") with Ray RLlib and RemoteRL,
explicitly specifying the AWS region to ensure consistency between local simulation
and remote training infrastructure.

Prerequisites:
- MuJoCo installed via Gymnasium: `pip install gymnasium[mujoco]`

Key concepts demonstrated:
- Defining and configuring RLlib's PPO algorithm (`PPOConfig`).
- Managing RLlib and SageMaker configurations using RemoteRL's `RemoteConfig`.
- Specifying AWS regions explicitly in simulation and training configurations.
- Simulating a MuJoCo environment locally to verify setup.
- Dynamically configuring SageMaker parameters prior to launching a cloud training job.

Dependencies:
- ray[rllib]
- remoterl
- gymnasium[mujoco]

Usage:
    python mujoco_gym_env_training.py
"""
from remoterl.remote_config import RemoteConfig
from ray.rllib.algorithms.ppo import PPOConfig


def main():
    # Initialize RLlib PPO algorithm configuration
    config = (
        PPOConfig()
        .env_runners(num_env_runners=4, num_envs_per_env_runner=32)
        .learners(num_learners=1, num_gpus_per_learner=1)
        .training(train_batch_size=1024, num_epochs=10, lr=5e-4)
    )

    # Initialize RemoteConfig with the RLlib algorithm configuration
    remote_config = RemoteConfig(config=config)

    # Simulate the MuJoCo environment locally specifying the AWS region explicitly
    remote_config.simulate(env="Walker2d-v5", region="ap-northeast-2")

    role_arn = input("Enter your SageMaker role ARN (or press enter to use dummy): ") or "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    output_path = input("Enter your S3 output path (or press enter to use dummy): ") or "s3://remoterl"

    if not output_path.startswith("s3://"):
        output_path = f"s3://{output_path}"
        print("s3 Output path:", output_path)

    # Configure SageMaker parameters explicitly specifying the AWS region
    remote_config.sagemaker(
        role_arn=role_arn,
        output_path=output_path,
        region="ap-northeast-2",
    )

    # Launch the training job on AWS SageMaker
    training_result = remote_config.train()

    print("Training job submitted. Result:", training_result)


if __name__ == "__main__":
    main()