# Standard library imports
import os
import platform
import subprocess
from typing import List
import argparse
import typer

def launch_simulator(args: List[str]) -> bool:
    env = os.environ.copy()
    simulation_script = os.path.join(os.path.dirname(__file__), "remote_simulator.py")
    system = platform.system()
    
    cmd_parts = ["python3", simulation_script] + args
    cmd_str = " ".join(cmd_parts)

    try:
        if system == "Linux":
            if not os.environ.get("DISPLAY"):
                subprocess.Popen(cmd_parts, env=env)
            else:
                try:
                    subprocess.Popen(
                        ['gnome-terminal', '--', 'bash', '-c', f'{cmd_str}; exec bash'],
                        env=env
                    )
                except FileNotFoundError:
                    subprocess.Popen(
                        ['xterm', '-e', f'{cmd_str}; bash'],
                        env=env
                    )
        elif system == "Darwin":
            apple_script = (
                'tell application "Terminal"\n'
                f'  do script "{cmd_str}"\n'
                '  activate\n'
                'end tell'
            )
            subprocess.Popen(['osascript', '-e', apple_script], env=env)
        elif system == "Windows":
            cmd_parts = ["python", simulation_script] + args
            cmd_str = " ".join(cmd_parts)
            cmd = f'start cmd /k "{cmd_str}"'
            subprocess.Popen(cmd, shell=True, env=env)
        else:
            typer.echo("Unsupported OS for launching a new terminal session.")
            return False
        return True
    except Exception as e:
        typer.echo(f"Failed to launch simulator: {e}")
        return False

def launch_simulation_from_config(env, num_env_runners, num_envs_per_runner, entry_point, region):
    from .cli.config import load_config, save_config, ensure_config_exists, wait_for_config_update, get_nested_config
    from .utils.connection import connect_to_remote_rl_server

    ensure_config_exists()

    config_data = load_config()

    config_data['rllib'].update({
        "env": env,
        "num_env_runners": num_env_runners,
        "num_envs_per_env_runner": num_envs_per_runner,
        "entry_point": entry_point,
    })   
    config_data['sagemaker'].update({
        "region": region,
    })   

    save_config(config_data)

    remote_training_key = connect_to_remote_rl_server(
        region=region,
        env_config={
            "env_id": env,
            "num_envs": num_env_runners,
        }
    )

    simulation_terminal = launch_simulator([
        "--remote_training_key", remote_training_key,
        "--region", region,
        "--env", env,
        "--entry_point", entry_point,
        "--num_env_runners", str(num_env_runners),
    ])

    if simulation_terminal is None:
        raise RuntimeError("Simulator subprocess failed to launch.")

    typer.echo("Simulation started. Please monitor the new window for logs.")

    try:
        updated_config = wait_for_config_update(remote_training_key, timeout=10)
        received_remote_training_key = get_nested_config(updated_config, "rllib", "remote_training_key")
        typer.echo(f"**Remote Training Key:** {received_remote_training_key}\n")
    except TimeoutError:
        typer.echo("Timeout occurred. Terminating simulation...")
        if simulation_terminal is not None:
            simulation_terminal.terminate()
            simulation_terminal.wait()
        raise

    return remote_training_key

def launch_all_env_servers(remote_training_key, remote_rl_server_url, 
                           env, num_env_runners, 
                           entry_point=None, env_dir=None):
    from remoterl.server.launcher import EnvLauncher
    
    launchers = [
        EnvLauncher.launch(
            remote_training_key,
            remote_rl_server_url,
            env,
            env_idx,
            entry_point=entry_point,
            env_dir=env_dir
        )
        for env_idx in range(num_env_runners)
    ]

    return launchers

def main():
    from remoterl.cli.config import load_config, save_config, ensure_config_exists
    from remoterl.utils.connection import get_remote_rl_server_url, connect_to_remote_rl_server
    is_docker_inside = bool(os.getenv("REMOTERL_CONFIG_PATH"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--num_env_runners", type=int, required=True)
    parser.add_argument("--entry_point", default=None)
    parser.add_argument("--remote_training_key", default=None)
    args = parser.parse_args()
    region = args.region
    env = args.env
    entry_point = args.entry_point
    num_env_runners = args.num_env_runners
    remote_training_key = args.remote_training_key
    
    if not remote_training_key:
        remote_training_key = connect_to_remote_rl_server(region=region)
    
    remote_rl_server_url = get_remote_rl_server_url(region)
    
    if not is_docker_inside:
        ensure_config_exists()

    launchers = launch_all_env_servers(
        remote_training_key=remote_training_key,
        remote_rl_server_url=remote_rl_server_url,
        env=env,
        num_env_runners=num_env_runners,
        entry_point = entry_point,
        env_dir = None
    )
    
    if not is_docker_inside:
        config_data = load_config()
        config_data.setdefault("rllib", {}).update({"remote_training_key": remote_training_key})
        save_config(config_data)

    typer.echo("Simulation running. Press Ctrl+C to terminate.")

    try:
        while any(l.server_thread.is_alive() for l in launchers):
            for launcher in launchers:
                launcher.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Termination requested. Stopping all servers...")
        for launcher in launchers:
            launcher.shutdown()
            launcher.server_thread.join(timeout=2)
    
    if not is_docker_inside:
        config_data = load_config()
        current_key = config_data.get("rllib", {}).get("remote_training_key")

        if current_key == remote_training_key:
            config_data["rllib"]["remote_training_key"] = None
            save_config(config_data)

    typer.echo("Simulation terminated successfully.")

if __name__ == "__main__":
    main()
