# remote_tune.py
import gymnasium
# Set the Gymnasium logger to only show errors
gymnasium.logger.min_level = gymnasium.logger.WARN
gymnasium.logger.warn = lambda *args, **kwargs: None
from ray.tune.registry import _global_registry, ENV_CREATOR
from gymnasium import spaces
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

class CustomGymEnv:
    def __init__(self, env, **kwargs):
        # `env` can be a single environment or a list of environments.
        self.env = env
        self.observation_space = kwargs.get("observation_space")
        self.action_space = kwargs.get("action_space")
        
    @classmethod
    def make(cls, name: str, **kwargs):
        env = gymnasium.make(name, **kwargs)
        return cls(
            env,
            observation_space=env.observation_space,
            action_space=env.action_space
        )

    @classmethod
    def make_vec(cls, name: str, num_envs, **kwargs):
        envs = [gymnasium.make(name, **kwargs) for _ in range(num_envs)]
        # Expand the observation and action spaces based on the first environment.
        obs_space = expand_space(envs[0].observation_space, num_envs)
        act_space = expand_space(envs[0].action_space, num_envs)
        return cls(envs, observation_space=obs_space, action_space=act_space)
    
    def reset(self, **kwargs):
        if isinstance(self.env, list):
            observations, infos = [], []
            for e in self.env:
                obs, info = e.reset(**kwargs)
                observations.append(obs)
                # infos.append(info)
            observations = np.array(observations)
            infos = {}
            return observations, infos
        else:
            return self.env.reset(**kwargs)

    def step(self, action):
        # In vectorized mode, `action` is assumed to be an iterable of actions,
        # one for each environment.
        if isinstance(self.env, list):
            observations, rewards, terminations, truncations = [], [], [], []
            for e, act in zip(self.env, action):
                obs, rew, terminated, truncated, info = e.step(act)
                observations.append(obs)
                rewards.append(rew)
                terminations.append(terminated)
                truncations.append(truncated)
            observations = np.array(observations)
            rewards = np.array(rewards)
            terminations = np.array(terminations)
            truncations = np.array(truncations)
            infos = {}
            print("Observations:", observations.shape)
            return observations, rewards, terminations, truncations, infos
        else:
            return self.env.step(action)

    def close(self):
        if isinstance(self.env, list):
            for e in self.env:
                e.close()
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