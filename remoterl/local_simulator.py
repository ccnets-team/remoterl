#remoterl/local_simulator.py
import argparse
import typer
import os
import platform
import subprocess
from typing import List

def launch_simulator(
    args: List[str]) -> subprocess.Popen:
    env = os.environ.copy()
    simulation_script = os.path.join(os.path.dirname(__file__), "local_simulator.py")
    system = platform.system()
    
    if system == "Linux":
        cmd_parts = ["python3", simulation_script] + args
        if not os.environ.get("DISPLAY"):
            # Headless mode: run in background without opening a terminal emulator.
            proc = subprocess.Popen(cmd_parts, env=env)
        else:
            cmd_str = " ".join(cmd_parts)
            try:
                proc = subprocess.Popen(
                    ['gnome-terminal', '--', 'bash', '-c', f'{cmd_str}; exec bash'],
                    env=env
                )
            except FileNotFoundError:
                proc = subprocess.Popen(
                    ['xterm', '-e', f'{cmd_str}; bash'],
                    env=env
                )
    elif system == "Darwin":
        cmd_parts = ["python3", simulation_script] + args
        cmd_str = " ".join(cmd_parts)
        apple_script = (
            'tell application "Terminal"\n'
            f'  do script "{cmd_str}"\n'
            '  activate\n'
            'end tell'
        )
        proc = subprocess.Popen(['osascript', '-e', apple_script], env=env)
    elif system == "Windows":
        cmd_parts = ["python", simulation_script] + args
        cmd_str = " ".join(cmd_parts)
        cmd = f'start cmd /k "{cmd_str}"'
        proc = subprocess.Popen(cmd, shell=True, env=env)
    else:
        typer.echo("Unsupported OS for launching a new terminal session.")
        raise typer.Exit(code=1)

    return proc

from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import importlib
def env_creator_from_entry_point(entry_point: str):
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    env_cls = getattr(module, class_name)

    def creator(config: EnvContext):
        return env_cls(config)

    return creator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_training_key", required=True)
    parser.add_argument("--remote_rl_server_url", required=True)
    
    args = parser.parse_args()
    
    remote_training_key = args.remote_training_key
    remote_rl_server_url = args.remote_rl_server_url
    
    from remoterl.utils.config_utils import load_config, save_config
    config_data = load_config()
    
    rllib_dict = config_data.get("rllib", {})
    num_env_runners = rllib_dict.get("num_env_runners")
    num_envs_per_env_runner = rllib_dict.get("num_envs_per_env_runner")

    env_type = rllib_dict.get("env_type")
    env_id = rllib_dict.get("env_id")
    entry_point = rllib_dict.get("entry_point")
    env_dir = rllib_dict.get("env_dir")
    
    if entry_point:
        if env_type == "gym":
            from remoterl.wrappers.gym_env import GymEnv
            GymEnv.register(env_id, entry_point)
        elif env_type == "rllib":
            env_creator = env_creator_from_entry_point(entry_point)
            register_env(env_id, env_creator)
    elif env_dir:
        if env_type == "unity":
            from remoterl.wrappers.unity_env import UnityEnv
            UnityEnv.register(env_id, entry_point)
                                
    launchers = []
    from remoterl.env_host.server import EnvServer
    for i in range(num_env_runners):
        env_idx = i
        launchers.append(
            EnvServer.launch(
                remote_training_key,
                remote_rl_server_url,
                env_type,
                env_id,
                env_idx,
                num_envs_per_env_runner,
            )
        )

    rllib_dict["remote_training_key"] = remote_training_key  # fixed plural naming consistency
    save_config(config_data)

    typer.echo("Simulation running. This terminal is now dedicated to simulation;")
    typer.echo("Press Ctrl+C to terminate the simulation.")

    try:
        while any(launcher.server_thread.is_alive() for launcher in launchers):
            for launcher in launchers:
                launcher.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Shutdown requested, stopping all servers...")
        for launcher in launchers:
            launcher.shutdown()
        for launcher in launchers:
            launcher.server_thread.join(timeout=2)

    # Cleanup after simulation ends
    config_data = load_config()
    if remote_training_key and config_data.get("rllib", {}).get("remote_training_key", {}) == remote_training_key:
        config_data["rllib"]["remote_training_key"] = None
    save_config(config_data)

    typer.echo("Simulation terminated.")

if __name__ == "__main__":
    main()
