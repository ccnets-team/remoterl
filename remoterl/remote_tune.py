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

def validate_array_elements(arr, valid_values):
    return np.all(np.isin(arr, valid_values))

def assert_valid_array(arr, valid_values):
    if not validate_array_elements(arr, valid_values):
        raise ValueError(f"Invalid array {arr}. Expected all elements to be in {valid_values}.")

class CustomGymEnv:
    def __init__(self, env, **kwargs):
        # `env` can be a single environment or a list of environments.
        self.env = env
        self.observation_space = kwargs.get("observation_space")
        self.action_space = kwargs.get("action_space")
        self.num_envs = kwargs.get("num_envs", 1)
        
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
        envs = gymnasium.make_vec(name, num_envs, **kwargs)
        obs_space = envs.observation_space  
        act_space = envs.action_space
        return cls(envs, observation_space=obs_space, action_space=act_space, num_envs = num_envs)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            action = np.array(action, dtype=np.int32)
            action = action.reshape(self.action_space.shape)
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