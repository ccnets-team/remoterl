import gymnasium as gym
from gymnasium.envs import registration

class CustomEnv(gym.Env):
    """
    A custom environment class that wraps around a Gymnasium environment.
    Replace or modify each method as needed to adapt to your specific simulator
    or custom environment logic.
    """

    def __init__(self, env, **kwargs):
        """
        Initialize the backend with a provided environment object.

        For your own custom environment:
        --------------------------------
        - You might directly initialize your environment's state here
          (instead of wrapping around `env`).
        - For example, if you have a custom simulator class, store it in
          `self.sim = MyCustomSimulator(...)` instead of `self.env`.
        - Make sure to define `self.observation_space` and `self.action_space`
          in a way that aligns with your environment's spaces.
        """
        self.env = env
        # Store these so the "observation_space()" and "action_space()" methods
        # can return them below. For a fully custom environment, define them
        # explicitly (e.g., Box, Discrete, MultiDiscrete, etc.).
        self.observation_space_ = self.env.observation_space
        self.action_space_ = self.env.action_space
        
    @staticmethod
    def make(env_id, **kwargs):
        """
        Create a single environment from a given ID (usually a Gymnasium-registered ID).

        For your own custom environment:
        --------------------------------
        - This method could create and return an instance of your
          environment class. For instance:
            return MyCustomSimulator(**kwargs)
        - Or if you're using a non-standard way to instantiate the env,
          just replace this logic with your custom creation process.
        """
        return gym.make(env_id, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """
        Create a vectorized environment, allowing multiple env copies to run in parallel.

        For your own custom environment:
        --------------------------------
        - If you support vectorized environments (like SubprocVectorEnv or AsyncVectorEnv),
          replace this call with logic that instantiates and returns a vector of custom envs.
        - If you don't need vectorized envs, you can remove or leave this unimplemented.
        """
        return gym.make_vec(env_id, num_envs=num_envs, **kwargs)

    def reset(self, **kwargs):
        """
        Reset the environment to its initial state.

        For your own custom environment:
        --------------------------------
        - Implement the logic to reset any internal state
          (positions, velocities, random seeds, etc.).
        - Return the initial observation and any optional info dict.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Take one step in the environment given an action.

        For your own custom environment:
        --------------------------------
        - Implement your simulator's step logic:
          1) Apply the action.
          2) Advance the simulation.
          3) Return (observation, reward, done, info).
        """
        observations, rewards, dones, infos = self.env.step(action)
        infos = {}
        final_observations = []
        infos['final_observation'] = final_observations        
        
        return observations, rewards, dones, infos

    def close(self):
        """
        Clean up and close the environment (e.g., stop rendering, shut down engines).

        For your own custom environment:
        --------------------------------
        - Add any cleanup logic, file I/O, or final logging needed
          when the environment is closed.
        """
        self.env.close()

    def observation_space(self):
        """
        Return the observation space object.

        For your own custom environment:
        --------------------------------
        - If you're not simply wrapping an existing Gym env, define
          self.observation_space_ as an instance of gym.spaces (Box, Discrete, etc.)
          in your constructor or setup method.
        """
        return self.observation_space_
    
    def action_space(self):
        """
        Return the action space object.

        For your own custom environment:
        --------------------------------
        - Same idea as observation_space(). Make sure it accurately matches
          the types of actions your environment expects.
        """
        return self.action_space_
    
    @classmethod
    def register(cls, id, entry_point):
        """
        Register a new environment under Gymnasium's registry system.

        For your own custom environment:
        --------------------------------
        - You can use this to dynamically register new env IDs,
          possibly pointing to a different entry point (class or function).
        - Adjust the printed message or remove it entirely as needed.
        """
        if id is None or entry_point is None:
            return  # Skip if both are None
        
        print(f"Registering environment: {id} with API URL: {entry_point}")

        registration.register(
            id=id,
            entry_point=entry_point,
            # You can pass additional kwargs here, which your env's constructor uses.
            kwargs={"entry_point": entry_point, "id": id}
        )
