import numpy as np
from gymnasium import Env
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityEnv(Env):
    # Class-level attribute to track instance count
    _instance_count = 0    
    env_dir = None
    env_id = None

    @classmethod
    def register(cls, env_id, env_entry_point, env_dir):
        """
        Register a Unity environment by setting the class-level entry_point and env_id.
        If the environment is already registered with the same values, skip updating.
        Otherwise, update the registration with the new values.
        """
        if cls.env_dir:
            if cls.env_dir != env_dir or cls.env_id != env_id:
                cls.env_dir = env_dir
                cls.env_id = env_id
                print(f"Updating registration for Unity environment: {env_id} with file path: {env_dir}")
            else:
                print(f"Unity environment {env_id} already registered with the same file path; skipping registration.")
        else:
            cls.env_dir = env_dir
            cls.env_id = env_id
            print(f"Registering Unity environment: {env_id} with file path: {env_dir}")
            
    @staticmethod
    def make(env_id, **kwargs):
        return UnityEnv(env_id=env_id, is_vectorized=False, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        return UnityEnv(env_id=env_id, num_envs=num_envs, is_vectorized=True, **kwargs)
        
    def __init__(self, env_id, num_envs=1, is_vectorized=False, **kwargs):
        """
        Initialize a Unity environment.

        :param env_id: Path to the Unity environment binary.
        :param num_envs: Number of environments (for vectorized environments).
        :param worker_id: Unique identifier for multiple Unity instances.
        :param use_graphics: Whether to run with graphics.
        :param is_vectorized: Whether the environment is vectorized.
        :param seed: Random seed for the environment.
        :param time_scale: Time scale for the Unity environment.
        """
        super().__init__()
        self.seed = UnityEnv._instance_count
        UnityEnv._instance_count += num_envs
        
        use_graphics = kwargs.get("use_graphics", False)
        time_scale = kwargs.get("time_scale", 4)
        if UnityEnv.env_dir is None:
            raise ValueError("Unity environment not registered. Use UnityEnv.register() to register the environment.")
        else:
            if env_id != UnityEnv.env_id:
                print(f"Environment ID mismatch: {env_id} != {UnityEnv.env_id}")
            
        self.env_id = env_id
        
        self.num_envs = num_envs
        self.is_vectorized = is_vectorized 
        self.no_graphics = not use_graphics
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(width=1280, height=720, time_scale=time_scale)

        if is_vectorized:
            # Create multiple environments without graphics for performance
            self.envs = [
                self.create_unity_env(
                    channel=self.channel,
                    no_graphics=True,
                    seed=self.seed + i,
                    worker_id=self.seed + i
                ) for i in range(num_envs)
            ]
        else:
            self.env = self.create_unity_env(
                channel=self.channel,
                no_graphics=self.no_graphics,
                seed=self.seed + 100,
                worker_id=self.seed + 100
            )
            self.envs = [self.env]  # For consistency, make self.envs a list

        self.behavior_names = []
        self.specs = []
        self.agent_per_envs = [] 
        self.from_local_to_global = []
        self.decision_agents = []
        self.num_agents = 0
        self._initialize_env_info()
        self._define_observation_space()
        self._define_action_space()
        
    @staticmethod
    def create_unity_env(channel, no_graphics, seed, worker_id):
        base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT
        env = UnityEnvironment(
            file_name=UnityEnv.env_dir,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[channel],
            worker_id=worker_id,
        )        
        return env

    def _initialize_env_info(self):
        total_agents = 0
        # Collect behavior names, specs, and agent counts for each environment
        for env in self.envs:
            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]
            self.behavior_names.append(behavior_name)
            self.specs.append(env.behavior_specs[behavior_name])
            # Get initial agent IDs and count
            # decision_steps, _ = env.get_steps(behavior_name)
            n_agents = len(env._env_state[behavior_name][0])
            self.agent_per_envs.append(n_agents)
            # env.reset()  # Reset the environment again before starting the episode

            self.decision_agents.append(np.zeros(n_agents, dtype=np.bool_))

            # Create mapping from local to global indices
            local_to_global = []
            for local_idx in range(n_agents):
                global_idx = total_agents + local_idx
                local_to_global.append(global_idx)
            self.from_local_to_global.append(local_to_global)

            total_agents += n_agents

        self.num_agents = total_agents

    def _define_observation_space(self):
        # Check consistency of observation shapes across all specs
        reference_shapes = [obs_spec.shape for obs_spec in self.specs[0].observation_specs]
        for spec in self.specs[1:]:
            current_shapes = [obs_spec.shape for obs_spec in spec.observation_specs]
            if reference_shapes != current_shapes:
                raise ValueError("Observation shapes are inconsistent across specs.")

        # Define the observation space per agent
        observation_shapes = reference_shapes  # Use the consistent shapes from the first spec

        # Check if any observation shape is an image
        if any(len(shape) == 3 for shape in observation_shapes):
            raise ValueError("Image observations are not supported.")

        # Since the assumption of RayPerceptionSensor is removed,
        # default to an infinite observation range for all observations
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, *obs_spec.shape)
                        if isinstance(obs_spec.shape, tuple) 
                        else (self.num_agents, obs_spec.shape),
                    dtype=np.float32
                )
                for obs_spec in self.specs[0].observation_specs
            ]
        )
        self.observation_shapes = observation_shapes

    def _define_action_space(self, start=1):
        # Ensure consistency of action specifications across all specs
        reference_action_spec = self.specs[0].action_spec
        for spec in self.specs[1:]:
            if spec.action_spec != reference_action_spec:
                raise ValueError("Action specifications are inconsistent across specs.")

        # Assume all environments have the same action space
        self.spec = self.specs[0]  # Use the first spec as the reference
        action_spec = self.spec.action_spec        

        # Define the action space per agent
        if action_spec.continuous_size > 0 and action_spec.discrete_size == 0:
            # Continuous actions only
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
        elif action_spec.discrete_size > 0 and action_spec.continuous_size == 0:
            # Discrete actions only
            if action_spec.discrete_size == 1:
                # Single discrete action branch
                self.action_space = spaces.Discrete(
                    action_spec.discrete_branches[0] - start, start=start
                )
            else:
                # Multiple discrete action branches
                self.action_space = spaces.MultiDiscrete(action_spec.discrete_branches - start, start=start)
        elif action_spec.continuous_size > 0 and action_spec.discrete_size > 0:
            # Mixed actions: Combine continuous and discrete spaces using Tuple
            continuous_space = spaces.Box(
                low=-1,
                high=1,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
            if action_spec.discrete_size == 1:
                discrete_space = spaces.Discrete(
                    action_spec.discrete_branches[0] - start, start=start
                )
            else:
                discrete_space = spaces.MultiDiscrete(action_spec.discrete_branches - start, start=start)
            self.action_space = spaces.Tuple((continuous_space, discrete_space))
        else:
            raise NotImplementedError("Action space configuration not supported.")
    
    def _create_action_tuple(self, actions, env_idx):
        action_tuple = ActionTuple()
        num_agents = len(actions)

        if num_agents == 0:
            # No agents to act upon
            return action_tuple

        if isinstance(self.action_space, spaces.Box):
            # Continuous actions only
            actions = np.asarray(actions, dtype=np.float32).reshape(num_agents, -1)
            action_tuple.add_continuous(actions)
        elif isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
            # Discrete actions only
            actions = np.asarray(actions, dtype=np.int32).reshape(num_agents, -1)
            action_tuple.add_discrete(actions)
        elif isinstance(self.action_space, spaces.Tuple):
            # Mixed actions: actions are tuples (continuous_action, discrete_action)
            continuous_actions = []
            discrete_actions = []
            continuous_action, discrete_action = actions
            continuous_action = np.asarray(continuous_action, dtype=np.float32).reshape(num_agents, -1)
            discrete_action = np.asarray(discrete_action, dtype=np.int32).reshape(num_agents, -1)

            action_tuple.add_continuous(continuous_actions)
            action_tuple.add_discrete(discrete_actions)
        else:
            raise NotImplementedError("Action type not supported.")

        return action_tuple
    
    def init_transitions(self, obs_len):

        num_agents = self.num_agents
        
        observations = tuple([[None] * num_agents for _ in range(obs_len)]) 
        final_observations = tuple([[None] * num_agents for _ in range(obs_len)]) 
        
        # Initialize other transition variables
        rewards = [None] * num_agents
        terminated = [None] * num_agents
        truncated = [None] * num_agents

        return observations, rewards, terminated, truncated, final_observations    

            
    def reset(self, **kwargs):
        """
        Reset the Unity environment(s) and retrieve initial observations.
        :return: Initial aggregated observations and info dictionary.
        """
        obs_len = len(self.observation_shapes)
        observations = tuple([[None] * self.num_agents for _ in range(obs_len)]) 
        for env_idx, env in enumerate(self.envs):
            env.reset()
            behavior_name = self.behavior_names[env_idx]
            decision_steps, _ = env.get_steps(behavior_name)

            self.decision_agents[env_idx] = np.zeros_like(self.decision_agents[env_idx])
            if len(decision_steps.agent_id) == 0:
                # No agents to act upon
                continue
            self.decision_agents[env_idx][decision_steps.agent_id] = True
            obs = decision_steps.obs
            for idx, agent_id in enumerate(decision_steps.agent_id):
                global_idx = self.from_local_to_global[env_idx][agent_id]
                # Aggregate all observation components
                for i in range(obs_len):
                    observations[i][global_idx] = obs[i][idx]

        return observations, {}
    
    def step(self, actions):
        """
        Perform a step in the Unity environment(s).
        :param actions: Actions to take for all agents.
        :return: Tuple containing observations, rewards, terminated flags, truncated flags, and info.
        """
        action_offset = 0

        # Set actions for all environments
        for env_idx, env in enumerate(self.envs):
            num_agents_in_env = self.agent_per_envs[env_idx]
            env_actions = actions[action_offset:action_offset + num_agents_in_env]
            action_offset += num_agents_in_env

            decision_check = self.decision_agents[env_idx]
            dec_actions = env_actions[decision_check]
            if len(dec_actions) > 0:
                action_tuple = self._create_action_tuple(dec_actions, env_idx)
                env.set_actions(self.behavior_names[env_idx], action_tuple)
            self.decision_agents[env_idx].fill(False)
            env.step()

        obs_len = len(self.observation_shapes)
        observations, rewards, terminated, truncated, final_observations = self.init_transitions(obs_len)
        # Collect results from all environments
        for env_idx, env in enumerate(self.envs):
            decision_steps, terminal_steps = env.get_steps(self.behavior_names[env_idx])
            self.decision_agents[env_idx] = np.zeros_like(self.decision_agents[env_idx])
            if len(decision_steps.agent_id) == 0:
                # No agents to act upon
                continue
            self.decision_agents[env_idx][decision_steps.agent_id] = True

            # Get agent IDs and mapping from agent_id to local index
            decision_agent_id_to_local = {agent_id: idx for idx, agent_id in enumerate(decision_steps.agent_id)}
            terminal_agent_id_to_local = {agent_id: idx for idx, agent_id in enumerate(terminal_steps.agent_id)}

            # Agents present in both decision and terminal steps
            common_agent_ids = set(decision_steps.agent_id).intersection(terminal_steps.agent_id)

            # Agents only in decision steps
            decision_only_agent_ids = set(decision_steps.agent_id) - common_agent_ids

            # Agents only in terminal steps
            terminal_only_agent_ids = set(terminal_steps.agent_id) - common_agent_ids

            # Handle agents present in both decision and terminal steps
            dec_obs = decision_steps.obs
            term_obs = terminal_steps.obs
            for agent_id in common_agent_ids:
                dec_local_idx = decision_agent_id_to_local[agent_id]
                term_local_idx = terminal_agent_id_to_local[agent_id]
                global_idx = self.from_local_to_global[env_idx][agent_id]
                # Aggregate observations
                for i in range(obs_len):
                    final_observations[i][global_idx] = term_obs[i][term_local_idx]
                    observations[i][global_idx] = dec_obs[i][dec_local_idx]
                
                rewards[global_idx] = float(terminal_steps.reward[term_local_idx])
                if terminal_steps.interrupted[term_local_idx]:
                    truncated[global_idx] = True  # Adjust if necessary
                    terminated[global_idx] = False
                else:
                    terminated[global_idx] = True
                    truncated[global_idx] = False

            # Handle agents only in decision steps
            for agent_id in decision_only_agent_ids:
                dec_local_idx = decision_agent_id_to_local[agent_id]
                global_idx = self.from_local_to_global[env_idx][agent_id]
                for i in range(obs_len):
                    observations[i][global_idx] = dec_obs[i][dec_local_idx]
                rewards[global_idx] = float(decision_steps.reward[dec_local_idx])
                terminated[global_idx] = False
                truncated[global_idx] = False

            # Handle agents only in terminal steps
            for agent_id in terminal_only_agent_ids:
                term_local_idx = terminal_agent_id_to_local[agent_id]
                global_idx = self.from_local_to_global[env_idx][agent_id]
                for i in range(obs_len):
                    observations[i][global_idx] = term_obs[i][term_local_idx]
                rewards[global_idx] = float(terminal_steps.reward[term_local_idx])
                if terminal_steps.interrupted[term_local_idx]:
                    truncated[global_idx] = True  # Adjust if necessary
                    terminated[global_idx] = False
                else:
                    terminated[global_idx] = True
                    truncated[global_idx] = False
   
        info = {}
        info['final_observation'] = final_observations
            
        return observations, rewards, terminated, truncated, info

    def close(self):
        """
        Close the Unity environment(s).
        """
        for env in self.envs:
            env.close()
            
        self.envs = []
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.envs:
            self.close()