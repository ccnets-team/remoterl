# env_wrapper/gym_env.py
import gymnasium as gym

class GymEnv:
    def __init__(self, env, **kwargs):
        """
        Initialize the GymEnv wrapper.

        Args:
            env: The underlying Gymnasium environment for extra customization.
            **kwargs: Additional keyword arguments that can be stored for custom behavior.
                For example, 'your_attr' can be used to store custom configuration.
        """
        self.env = env                     # Underlying Gym environment
        self.kwargs = kwargs               # Save extra keyword arguments
        self.your_attr = kwargs.get("your_attr", {})  # Custom attribute for any extra data
        self.observation_space = env.observation_space  # Mirror the observation space of the underlying env
        self.action_space = env.action_space            # Mirror the action space of the underlying env

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensure that the environment is closed when exiting a context.
        """
        self.close()

    # Factory methods to create environments with the custom wrapper
    @staticmethod
    def make(env_id, **kwargs):
        """
        Create a single environment wrapped by GymEnv.

        Args:
            env_id (str): The Gymnasium environment ID to create.
            **kwargs: Additional keyword arguments to pass to gym.make and store in the wrapper.

        Returns:
            GymEnv: An instance of GymEnv wrapping the created environment.
        """
        # Create the environment using Gymnasium's make() and wrap it in GymEnv.
        return GymEnv(gym.make(env_id, **kwargs), **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """
        Create a vectorized environment wrapped by GymEnv.

        Args:
            env_id (str): The Gymnasium environment ID.
            num_envs (int): The number of environments to vectorize.
            **kwargs: Additional keyword arguments to pass to gym.make_vec and store in the wrapper.

        Returns:
            GymEnv: An instance of GymEnv wrapping the vectorized environment.
        """
        # Create a vectorized environment and wrap it.
        return GymEnv(gym.make_vec(env_id, num_envs=num_envs, **kwargs), **kwargs)

    def reset(self, **kwargs):
        """
        Reset the underlying environment.

        Returns:
            The initial observation and any additional info provided by the env's reset.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Take a step in the underlying environment using the given action.

        Args:
            action: The action to be executed in the environment.

        Returns:
            A tuple (observation, reward, terminated, truncated, info) as returned by the env.
        """
        return self.env.step(action)

    def close(self):
        """
        Close the underlying environment and clean up resources.
        """
        if self.env:
            self.env.close()
            self.env = None

    @classmethod
    def register(cls, env_id, env_entry_point, env_dir):
        """
        Register an environment with Gymnasium's registry if it isn't already registered.

        Args:
            env_id (str): The unique environment ID to register.
            env_entry_point (str): The fully qualified entry point for the environment.
            env_dir (str): A directory parameter (if needed) for environment files; not used directly here.

        Behavior:
            - If the environment is already registered, a message is printed and registration is skipped.
            - Otherwise, it attempts to register using Gymnasium's registration.register.
        """
        from gymnasium.envs import registration
        from gymnasium.error import UnregisteredEnv  # For older versions, it might be gym.error.Error
        import gymnasium
        try:
            # Check if the environment is already registered.
            gymnasium.spec(env_id)
            print(f"Environment {env_id} is already registered; skipping registration.")
        except UnregisteredEnv:
            print(f"Registering Gym environment: {env_id} with entry_point: {env_entry_point}")
            try:
                registration.register(
                    id=env_id,
                    entry_point=env_entry_point,
                )
            except Exception as e:
                print(f"Error registering environment {env_id}: {e}")
                raise e

def is_gymnasium_envs(env_id: str):
    """
    Retrieves environment IDs grouped by specified categories based on entry points in the Gymnasium registry.
    
    Returns:
        dict: A dictionary where keys are categories (e.g., "classic_control", "box2d") and values are lists of 
        environment IDs for each category.
    """
    categories = ["classic_control", "box2d", "toy_text", "mujoco", "phys2d", "tabular"]        
    envs_by_category = {category: [] for category in categories}
    from gymnasium import envs
    
    for env_spec in envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            for category in categories:
                if category in env_spec.entry_point:
                    envs_by_category[category].append(env_spec.id)
                    break  # Move to the next env_spec after finding a match for the current category

    # Flatten the dictionary to check if the environment is in any category
    all_registered_envs = [env_id for env_list in envs_by_category.values() for env_id in env_list]
    if env_id in all_registered_envs:
        return True
    return False