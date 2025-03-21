
from typing import Dict
import json
import websocket
import time
from ..local_simulator import launch_simulator

def connect_to_remote_rl_server(region: str, env_config: Dict) -> str:
    
    ws = websocket.WebSocket()
    if region not in ["us-east-1", "ap-northeast-2"]:
        raise ValueError(f"Invalid region: {region}")
    
    remote_rl_server_url = f"wss://{region}.ccnets.org"
    
    ws.connect(remote_rl_server_url)
    
    register_request = json.dumps({
        "action": "register", 
        "data": env_config
    })
    ws.send(register_request)
    remote_training_key = ws.recv()
    ws.close()      
    
    return remote_rl_server_url, remote_training_key

def wait_for_config_update(sent_remote_training_key, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        from remoterl.utils.config_utils import load_config
        config_data = load_config()  # Your function to load the config file.
        registered_remote_training_key = config_data.get("rllib", {}).get("remote_training_key")
        if sent_remote_training_key == registered_remote_training_key:
            return config_data
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for config update.")

def do_simulation(env_type, env, num_envs_per_env_runner, num_env_runners, entry_point, env_dir, region):
    env_config = {
        "env_id": env,
        "num_envs": num_env_runners,
    }
    remote_rl_server_url, remote_training_key = connect_to_remote_rl_server(region, env_config)

    # Initial args as list (command-line-style)
    extra_args = [
        "--remote_training_key", remote_training_key,
        "--remote_rl_server_url", remote_rl_server_url,
        "--env_type", env_type,
        "--env_id", env,
        "--num_agents", str(num_env_runners * num_envs_per_env_runner),
        "--num_envs", str(num_env_runners),
    ]
    
    if entry_point:
        env_config["entry_point"] = entry_point
    if env_dir:
        env_config["env_dir"] = env_dir

    # Dynamically add other args from env_config
    for key, value in env_config.items():
        if value is not None:
            extra_args.extend([f"--{key}", str(value)])

    print("Starting the simulation in a separate terminal window. Please monitor that window for real-time logs.")
    
    simulation_terminal = launch_simulator(extra_args)
    
    try:
        updated_config = wait_for_config_update(remote_training_key, timeout=10)
        received_remote_training_key = updated_config["rllib"]["remote_training_key"]
        print("Remote Training Key for simulation updated successfully:")
        print("**Remote Training Key Under**\n")
        print(received_remote_training_key)
        print("Simulation is now running. Please run 'remoterl train' to continue..")
    except TimeoutError:
        print("Configuration update timed out. Terminating simulation process.")
        simulation_terminal.terminate()
        simulation_terminal.wait()
    
    return remote_training_key