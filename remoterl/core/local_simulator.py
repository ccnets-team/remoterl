# Standard library imports
import os
import platform
import subprocess
from typing import List
import argparse
import typer

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

def launch_remote_rl_simulation(env, num_env_runners, num_envs_per_runner, entry_point, region):
    # Delayed local imports to avoid circularity
    from ..cli.config import load_config, save_config, ensure_config_exists, wait_for_config_update, get_nested_config
    from ..utils.connection import connect_to_remote_rl_server

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
     
    remote_rl_server_url, remote_training_key = connect_to_remote_rl_server(
        region=region,
        env_config={
            "env_id": env,
            "num_envs": num_env_runners,
        }
    )

    simulation_terminal = launch_simulator([
        "--remote_training_key", remote_training_key,
        "--remote_rl_server_url", remote_rl_server_url,
    ])

    print("Simulation started. Please monitor the new window for logs.")

    try:
        updated_config = wait_for_config_update(remote_training_key, timeout=10)
        received_remote_training_key = get_nested_config(updated_config, "rllib", "remote_training_key")
        print(f"**Remote Training Key:** {received_remote_training_key}\n")
    except TimeoutError:
        print("Timeout occurred. Terminating simulation...")
        simulation_terminal.terminate()
        simulation_terminal.wait()

    return remote_training_key

def launch_all_env_servers(remote_training_key, remote_rl_server_url, 
                           env, num_env_runners, num_envs_per_runner,
                           entry_point=None, env_dir=None):
    from remoterl.server.launcher import EnvLauncher
    
    launchers = [
        EnvLauncher.launch(
            remote_training_key,
            remote_rl_server_url,
            env,
            env_idx,
            num_envs_per_runner,
            entry_point=entry_point,
            env_dir=env_dir
        )
        for env_idx in range(num_env_runners)
    ]

    return launchers

def main():
    from remoterl.cli.config import load_config, save_config, ensure_config_exists

    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_training_key", required=True)
    parser.add_argument("--remote_rl_server_url", required=True)
    args = parser.parse_args()
    
    ensure_config_exists()
    
    config_data = load_config()
    rllib_dict = config_data.get("rllib", {})

    num_env_runners = rllib_dict.get("num_env_runners")
    num_envs_per_env_runner = rllib_dict.get("num_envs_per_env_runner")
    env = rllib_dict.get("env")
    entry_point = rllib_dict.get("entry_point")
    env_dir = rllib_dict.get("env_dir")

    launchers = launch_all_env_servers(
        remote_training_key=args.remote_training_key,
        remote_rl_server_url=args.remote_rl_server_url,
        env=env,
        num_env_runners=num_env_runners,
        num_envs_per_runner=num_envs_per_env_runner,
        entry_point = entry_point,
        env_dir = env_dir
    )

    rllib_dict["remote_training_key"] = args.remote_training_key
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

    config_data = load_config()
    current_key = config_data.get("rllib", {}).get("remote_training_key")

    if current_key == args.remote_training_key:
        config_data["rllib"]["remote_training_key"] = None
        save_config(config_data)

    typer.echo("Simulation terminated successfully.")

if __name__ == "__main__":
    main()
