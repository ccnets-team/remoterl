from threading import Thread
from .env_api import EnvAPI
from ..remote_env import RemoteMultiAgentEnv

class EnvLauncher(EnvAPI):
    """
    EnvLauncher extends EnvAPI to facilitate the local hosting and launching of RLlib-compatible environments.

    This class provides simplified integration with remote reinforcement learning infrastructures, allowing users
    to initiate environment instances quickly for remote training sessions. It specifically leverages the RLlib
    environment registry and frameworks, with other environment types deprecated in this implementation.

    Args:
        remote_training_key (str): Unique key for linking to the remote training infrastructure.
        remote_rl_server_url (str): URL for the remote RL server connection.
        env_idx (int): Index to distinguish among multiple concurrent environment instances.
        num_agents (int): Number of parallel agents/environments to manage.

    Usage:
        launcher = EnvLauncher.launch(
            remote_training_key="your-key",
            remote_rl_server_url="ws://your-url",
            env_idx=0,
            num_agents=4
        )
    """
    def __init__(
        self,
        remote_training_key,
        remote_rl_server_url,
        env,
        env_idx,
        entry_point=None,
        env_dir=None
    ):
        if entry_point:
            from gymnasium import register
            register(env, entry_point)
                    
        super().__init__(RemoteMultiAgentEnv, remote_training_key, remote_rl_server_url, env_idx)

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
    def launch(cls, remote_training_key, remote_rl_server_url, env, env_idx,
               entry_point=None, env_dir=None) -> "EnvLauncher":
        instance = cls(
            remote_training_key=remote_training_key,
            remote_rl_server_url=remote_rl_server_url,
            env=env,
            env_idx=env_idx,
            entry_point=entry_point,
            env_dir=env_dir
        ) 
        instance.run_thread_server()
        return instance