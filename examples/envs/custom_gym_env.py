# custom_gym_env.py
from remoterl import remote_tune 
from remoterl.remote_config import RemoteConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.custom_gym_env import SimpleCorridor
from ray import tune

def main():
    # Initialize with an RLlib configuration.
    config = (
        PPOConfig()
        .env_runners(num_env_runners=4, num_envs_per_env_runner=16, rollout_fragment_length='auto', sample_timeout_s=60)
        .learners(num_learners=1, num_gpus_per_learner=1)             
        .training(train_batch_size=64, num_epochs=10, lr=1e-4)
    )
    remote_config = RemoteConfig(config=config)
    
    tune.register_env("corridor-env", lambda config: SimpleCorridor(config))
    
    # since remote rl runs parallel processes, we need to register the environment entry point and receive from the server
    entry_point = remote_tune.get_entry_point("corridor-env")
    
    # if env_type is None then it will default to custom gym environment 
    training_key = remote_config.simulate(env = "corridor-env", entry_point = entry_point)
    print("Remote Training Key:", training_key)
    
    print("configs: ", remote_config.to_dict())
    
if __name__ == "__main__":
    main()
