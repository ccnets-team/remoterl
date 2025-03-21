import gymnasium
# Set the Gymnasium logger to only show errors
gymnasium.logger.min_level = gymnasium.logger.WARN
gymnasium.logger.warn = lambda *args, **kwargs: None

from remoterl.remote_rl import RemoteRL
from ray.rllib.algorithms.ppo import PPOConfig


def main():
    # Initialize with an RLlib configuration.
    config = (
        PPOConfig()
        .env_runners(num_env_runners=4)   
        .learners(num_learners=1, num_gpus_per_learner=1)             
        .training(train_batch_size=2048, num_sgd_iter=10, lr=5e-4)
    )
    remote_rl = RemoteRL(config=config)
    
    # Run a simulation to obtain a remote training key.
    region = input("Enter region (or press enter to use us-east-1 or ap-northeast-2): ") or "us-east-1"
    training_key = remote_rl.simulate(region=region)
    print("Remote Training Key:", training_key)
    
    print("configs: ", remote_rl.rllib_config.to_dict())
    # Prompt for role ARN (with a dummy default for testing)
    role_arn = input("Enter your SageMaker role ARN (or press enter to use dummy): ") or "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    output_path = input("Enter your S3 output path (or press enter to use dummy): ") or "s3://remoterl"
    if not output_path.startswith("s3://"):
        output_path = f"s3://{output_path}"
        print("s3 Output path:", output_path)

    # Configure SageMaker parameters just before training.
    remote_rl.config_sagemaker(
        role_arn=role_arn,
        output_path=output_path,
        region=region
    )
    # Launch the training job on the cloud.
    training_result = remote_rl.train()
    print("Training job submitted. Result:", training_result)
    
if __name__ == "__main__":
    main()
