import gymnasium as gym
import copy

class RemoteMultiAgentEnv(gym.Env):
    """
    RemoteMultiAgentEnv is a wrapper around Gymnasium environments specifically designed for
    remote reinforcement learning training with Ray RLlib.

    It manages multiple Gymnasium environment instances concurrently, representing each instance
    as a distinct agent within a single RLlib-compatible multi-agent environment. This structure
    facilitates parallel environment interaction, significantly simplifying distributed
    reinforcement learning training.

    RemoteMultiAgentEnv provides native integration with RLlib's MultiAgentEnv API, allowing
    streamlined deployment and efficient remote training on cloud platforms, including AWS SageMaker.

    Args:
        config (dict, optional): RLlib's environment context providing configurations.
            - num_envs (int): Number of parallel environment instances to create.
            - env (str): The Gymnasium environment ID (e.g., "CartPole-v1").

    Methods:
        reset(seed=None, options=None):
            Reset all managed environments.

        step(action_dict):
            Execute a step in each environment based on the provided actions.

        close():
            Close all managed environments.

        make(env_id):
            Class method to create a single environment instance.

        make_vec(env_id, num_envs):
            Class method to create a vectorized multi-environment instance.
    """    
    def __init__(self, config: dict = None):
        super().__init__()

        if config is None:
            config = {}
        else:
            config = copy.deepcopy(config)

        num_envs = config.pop("num_envs", 1)
        env = config.pop("env", None)
        if env is None:
            raise ValueError("env must be provided.")

        self.envs = [gym.make(env) for _ in range(num_envs)]
        
        self.agents = [str(i) for i in range(len(self.envs))]
        self.possible_agents = self.agents.copy()

        self.observation_space = {agent_id: self.envs[i].observation_space for i, agent_id in enumerate(self.agents)}
        self.action_space = {agent_id: self.envs[i].action_space for i, agent_id in enumerate(self.agents)}

    def reset(self, seed=None, options=None):
        obs, infos = {}, {}
        for agent_id, env in zip(self.agents, self.envs):
            ob, info = env.reset(seed=seed, options=options)
            obs[agent_id] = ob
            infos[agent_id] = info

        return obs, infos

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        if isinstance(action_dict, dict):
            for agent_id, action in action_dict.items():
                idx = int(agent_id)  # explicitly convert agent_id to env index
                ob, reward, terminated, truncated, info = self.envs[idx].step(action)

                obs[agent_id] = ob
                rewards[agent_id] = reward
                terminateds[agent_id] = terminated
                truncateds[agent_id] = truncated
                infos[agent_id] = info
        else:
            for idx, action in enumerate(action_dict):
                if action is None:
                    continue
                agent_id = str(idx)
                ob, reward, terminated, truncated, info = self.envs[idx].step(action)

                obs[agent_id] = ob
                rewards[agent_id] = reward
                terminateds[agent_id] = terminated
                truncateds[agent_id] = truncated
                infos[agent_id] = info

        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        for env in self.envs:
            env.close()
    
    @classmethod
    def make(cls, env_id):
        env_context = {"env": env_id}
        return cls(env_context)

    @classmethod
    def make_vec(cls, env_id, num_envs):
        env_context = {"env": env_id, "num_envs": num_envs}
        return cls(env_context)

