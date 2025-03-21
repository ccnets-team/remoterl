
from remoterl.remote_config import RemoteConfig
from ray.rllib.algorithms.ppo import PPOConfig

def main():
    # Initialize with an RLlib configuration.
    config = (
        PPOConfig()
        .env_runners(num_env_runners=4, num_envs_per_env_runner=32)   
        .learners(num_learners=1, num_gpus_per_learner=1)             
        .training(train_batch_size=1024, num_epochs=10, lr=5e-4)
    )
    remote_config = RemoteConfig(config=config)
    
    remote_config.simulate(env= "Walker2d-v5")
    
    role_arn = input("Enter your SageMaker role ARN (or press enter to use dummy): ") or "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    output_path = input("Enter your S3 output path (or press enter to use dummy): ") or "s3://remoterl"
    
    if not output_path.startswith("s3://"):
        output_path = f"s3://{output_path}"
        print("s3 Output path:", output_path)

    # Configure SageMaker parameters just before training.
    remote_config.sagemaker(
        role_arn=role_arn,
        output_path=output_path,
    )
    # Launch the training job on the cloud.
    training_result = remote_config.train()
    
    print("Training job submitted. Result:", training_result)
    
if __name__ == "__main__":
    main()
