import argparse
import typer
import os
import platform
import subprocess
from typing import List

def open_simulation_in_screen(extra_args: List[str]) -> subprocess.Popen:
    env = os.environ.copy()
    simulation_script = os.path.join(os.path.dirname(__file__), "simulation.py")
    system = platform.system()

    if system == "Linux":
        cmd_parts = ["python3", simulation_script] + extra_args
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
        cmd_parts = ["python3", simulation_script] + extra_args
        cmd_str = " ".join(cmd_parts)
        apple_script = (
            'tell application "Terminal"\n'
            f'  do script "{cmd_str}"\n'
            '  activate\n'
            'end tell'
        )
        proc = subprocess.Popen(['osascript', '-e', apple_script], env=env)
    elif system == "Windows":
        cmd_parts = ["python", simulation_script] + extra_args
        cmd_str = " ".join(cmd_parts)
        cmd = f'start cmd /k "{cmd_str}"'
        proc = subprocess.Popen(cmd, shell=True, env=env)
    else:
        typer.echo("Unsupported OS for launching a new terminal session.")
        raise typer.Exit(code=1)
    return proc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_training_key", required=True)
    parser.add_argument("--remote_rl_server_url", required=True)
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--env_id", required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--num_agents", type=int, required=True)

    args = parser.parse_args()

    from remoterl.utils.config_utils import load_config, save_config
    from remoterl.env_host.server import EnvServer
    import typer

    config_data = load_config()

    remote_training_key = args.remote_training_key
    remote_rl_server_url = args.remote_rl_server_url
    env_type = args.env_type
    env_id = args.env_id
    num_envs = args.num_envs
    num_agents = args.num_agents
    base_agents, remainder = divmod(num_agents, num_envs)
    agents_per_env = [base_agents + (1 if i < remainder else 0) for i in range(num_envs)]
    
    launchers = []
    for i in range(num_envs):
        env_idx = i
        launchers.append(
            EnvServer.launch(
                remote_training_key,
                remote_rl_server_url,
                env_type,
                env_id,
                env_idx,
                agents_per_env[i],
            )
        )
    config_data["rllib"]["remote_training_key"] = remote_training_key  # fixed plural naming consistency
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
