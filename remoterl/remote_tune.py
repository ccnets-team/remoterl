# remote_tune.py
import gymnasium
# Set the Gymnasium logger to only show errors
gymnasium.logger.min_level = gymnasium.logger.WARN
gymnasium.logger.warn = lambda *args, **kwargs: None
from ray.tune.registry import _global_registry, ENV_CREATOR
from gymnasium import spaces
from typing import Optional, List, Any
import numpy as np

def expand_space(space, num_envs):
    if isinstance(space, spaces.Box):
        low = np.repeat(np.expand_dims(space.low, axis=0), num_envs, axis=0)
        high = np.repeat(np.expand_dims(space.high, axis=0), num_envs, axis=0)
        return spaces.Box(low=low, high=high, dtype=space.dtype)
    elif isinstance(space, spaces.Discrete):
        return space
    elif isinstance(space, spaces.MultiDiscrete):
        return space
    elif isinstance(space, spaces.Tuple):
        return spaces.Tuple([expand_space(sub, num_envs) for sub in space.spaces])
    else:
        # For other space types, you might just create a Tuple of the space repeated.
        return spaces.Tuple([space for _ in range(num_envs)])

def validate_array_elements(arr, valid_values):
    return np.all(np.isin(arr, valid_values))

def assert_valid_array(arr, valid_values):
    if not validate_array_elements(arr, valid_values):
        raise ValueError(f"Invalid array {arr}. Expected all elements to be in {valid_values}.")

class RLlibEnv:
    def __init__(self, env):
        # env can be a single env or multi-agent env
        self.env = env
        self.is_vectorized = isinstance(env, list)
        if self.is_vectorized:
            self.num_envs = len(env)
            self.single_observation_space = env[0].observation_space
            self.single_action_space = env[0].action_space 
            self.observation_space = expand_space(self.single_observation_space, self.num_envs)
            self.action_space = expand_space(self.single_action_space, self.num_envs)
        else:
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.num_envs = 1
        
    @classmethod
    def make(cls, env_id: str, **kwargs):
        env = gymnasium.make(env_id, **kwargs)
        return cls(env)

    @classmethod
    def make_vec(cls, env_id: str, num_envs: int, **kwargs):
        multi_agent_env = [gymnasium.make(env_id, **kwargs) for _ in range(num_envs)]
        return cls(multi_agent_env)

    def reset(self, seed=None, options=None):
        if not self.is_vectorized:
            obs, info = self.env.reset(seed=seed, options=options)
            return obs, info

        obs_batch, info_batch = [], []

        for idx in range(self.num_envs):
            obs, info = self.env[idx].reset(seed=seed, options=options)
            obs_batch.append(obs)
            info_batch.append(info)

        return np.array(obs_batch), info_batch

    def step(self, actions: List[Optional[Any]]):
        if not self.is_vectorized:
            return self.env.step(actions)

        assert len(actions) == self.num_envs, "The length of actions must match num_envs"

        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = [], [], [], [], []

        for idx in range(self.num_envs):
            if actions[idx] is not None:
                if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
                    actions[idx] = np.array(actions[idx], dtype=np.int32)
                    actions[idx] = actions[idx].reshape(self.single_action_space.shape)
                obs, reward, terminated, truncated, info = self.env[idx].step(actions[idx])
            else:
                continue

            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)

        return (
            np.array(obs_batch),
            np.array(reward_batch, dtype=np.float32),
            np.array(terminated_batch, dtype=bool),
            np.array(truncated_batch, dtype=bool),
            info_batch,
        )
        
    def close(self):
        if self.is_vectorized:
            for env in self.env:
                env.close()
        else:
            self.env.close()
    
    def register(name: str, entry_point: str):
        from gymnasium.error import UnregisteredEnv  # For older versions, it might be gym.error.Error
        import gymnasium
        try:
            # Check if the environment is already registered.
            gymnasium.spec(name)
            print(f"Environment {name} is already registered; skipping registration.")
        except UnregisteredEnv:
            print(f"Registering Gym environment: {name} with entry_point: {entry_point}")
            try:
                gymnasium.register(
                    id=name,
                    entry_point=entry_point,
                )
            except Exception as e:
                print(f"Error registering environment {name}: {e}")
                raise e        

def get_entry_point(name: str):
    env_creator = _global_registry.get(ENV_CREATOR, name)
    if not env_creator:
        return None
    try:
        env_instance = env_creator({})
    except Exception as e:
        env_instance = env_creator()
    entry_point = f"{env_instance.__class__.__module__}:{env_instance.__class__.__name__}"
    return entry_point