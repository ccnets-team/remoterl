from threading import Thread
from .env_api import EnvAPI

class EnvServer(EnvAPI):
    """
    EnvServer extends EnvAPI to manage environment hosting locally.
    It integrates the launching functionality so that you can simply call
    EnvServer.launch(...) to start a server.
    """
    def __init__(
        self,
        remote_training_key,
        remote_rl_server_url,
        env_type,
        env_id,
        env_idx,
        num_agents,
    ):
        if env_type.lower() in ["gym", "gymnasium", "mujoco"]:
            from ..wrappers.gym_env import GymEnv, is_gymnasium_envs
            env_wrapper = GymEnv

            # Retrieve all registered MuJoCo environment IDs explicitly
            if is_gymnasium_envs(env_id):
                try:
                    import mujoco
                except ImportError:
                    raise ImportError(
                        f"Environment '{env_id}' requires MuJoCo. "
                        "Please install it via: pip install 'gymnasium[mujoco]'"
                    )
                   
        elif env_type.lower() in ["unity", "mlagents"]:
            try:
                import mlagents_envs
                import google.protobuf
            except ImportError as e:
                raise ImportError("Required packages for Unity environment are missing: " + str(e))
            
            if not mlagents_envs.__version__.startswith("0.30"):
                raise ImportError(f"mlagents_envs version 0.30 is required, but found {mlagents_envs.__version__}")
            if not google.protobuf.__version__.startswith("3.20"):
                raise ImportError(f"protobuf version 3.20 is required, but found {google.protobuf.__version__}")
            
            from ..wrappers.unity_env import UnityEnv
            env_wrapper = UnityEnv

        else:
            if not (env_type.lower() in ["rllib", "custom", "custom_gym"]):
                print(f"Unknown environment type: {env_type}. Defaulting to 'custom' environment.")
            from ..remote_env import RemoteMultiAgentEnv
            env_wrapper = RemoteMultiAgentEnv
            
        super().__init__(env_wrapper, remote_training_key, remote_rl_server_url, env_idx, num_agents)

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return super().__exit__(exc_type, exc_value, traceback)

    def run_thread_server(self):
        """Run the server in a separate daemon thread with a graceful shutdown mechanism."""
        self.server_thread = Thread(target=self.communicate, daemon=True)
        self.server_thread.start()
        return self.server_thread

    def shutdown(self):
        """Signal the server to shut down gracefully."""
        self.shutdown_event.set()

    @classmethod
    def launch(cls, remote_training_key, remote_rl_server_url, 
               env_type, env_id,
               env_idx, num_agents) -> "EnvServer":
        instance = cls(
            remote_training_key=remote_training_key,
            remote_rl_server_url=remote_rl_server_url,
            env_type=env_type,
            env_id=env_id,
            env_idx=env_idx,
            num_agents=num_agents,
        ) 
        instance.run_thread_server()
        return instance