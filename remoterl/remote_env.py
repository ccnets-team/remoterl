import copy
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext

class RemoteMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: EnvContext = None):
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
        return cls(EnvContext({"env": env_id}, worker_index=0))

    @classmethod
    def make_vec(cls, env_id, num_envs):
        return cls(EnvContext({"env": env_id, "num_envs": num_envs}, worker_index=0))

