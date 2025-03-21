
from remoterl import remote_tune 
from remoterl.remote_config import RemoteConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.custom_gym_env import SimpleCorridor

def main():
    # Initialize with an RLlib configuration.
    config = (
        PPOConfig()
        .env_runners(num_env_runners=4, num_envs_per_env_runner=32)   
        .learners(num_learners=1, num_gpus_per_learner=1)             
        .training(train_batch_size=1024, num_epochs=10, lr=5e-4)
    )
    remote_config = RemoteConfig(config=config)
    
    # tune.register_env("corridor-env", lambda config: SimpleCorridor(config))
    remote_tune.register_env("corridor-env", lambda config: SimpleCorridor(config))
    
    # if env_type is None then it will default to custom gym environment 
    training_key = remote_config.simulate()
    print("Remote Training Key:", training_key)
    
    print("configs: ", remote_config.to_dict())
    
if __name__ == "__main__":
    main()
