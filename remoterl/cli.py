import warnings
import logging

# Suppress specific pydantic warning about the "json" field.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "json" in "MonitoringDatasetFormat" shadows an attribute in parent "Base"',
    category=UserWarning,
    module="pydantic._internal._fields"
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

import typer
import os
import re
import time
import yaml
import json
import websocket
from .core import RemoteRL
from .config.sagemaker import SageMakerConfig
from .config.rllib import RLLibConfig
from typing import Optional, Dict
import requests 
from .simulation import open_simulation_in_screen
from .utils.config_utils import load_config, save_config, generate_default_section_config, update_config_using_method, ensure_config_exists
from .utils.config_utils import convert_to_objects, parse_extra_args, update_config_by_dot_notation
from .utils.config_utils import DEFAULT_CONFIG_PATH, TOP_CONFIG_CLASS_MAP

app = typer.Typer(add_completion=False, invoke_without_command=True)

def load_help_texts(yaml_filename: str) -> Dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, yaml_filename)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def auto_format_help(text: str) -> str:
    formatted = re.sub(r'([.:])\s+', r'\1\n\n', text)
    return formatted

help_texts = load_help_texts("help_config.yaml")
    
@app.command(
    "config",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    short_help=help_texts["config"]["short_help"],
    help=auto_format_help(help_texts["config"]["detailed_help"]),
)
def config(ctx: typer.Context):
    ensure_config_exists()
    
    if not ctx.args:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    current_config = load_config()
    config_obj = convert_to_objects(current_config)

    # Decide the mode based on the first argument.
    if ctx.args[0].startswith("--"):
        new_changes = parse_extra_args(ctx.args)
        update_log = update_config_by_dot_notation(config_obj, new_changes)
    else:
        update_log = update_config_using_method(ctx.args, config_obj)

    # Print detailed change summaries.
    for key, old_value, new_value, changed, message in update_log:
        if changed:
            method_configuration = True if old_value is None and new_value is None else False
            if method_configuration:
                typer.echo(typer.style(
                    f" - {key} {message}.",
                    fg=typer.colors.GREEN
                ))
            else:
                typer.echo(typer.style(
                    f" - {key} changed from {old_value} to {new_value}",
                    fg=typer.colors.GREEN
                ))
        else:
            typer.echo(typer.style(
                f" - {key}: no changes applied because {message}",
                fg=typer.colors.YELLOW
            ))
            
    full_config = {key: obj.to_dict() for key, obj in config_obj.items()}
    save_config(full_config)

@app.command(
    "edit",
    short_help=help_texts["edit"]["short_help"],
    help=auto_format_help(help_texts["edit"]["detailed_help"]),
)
def edit_config():   
    ensure_config_exists()
    try:
        import platform
        import subprocess
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["notepad.exe", DEFAULT_CONFIG_PATH])
        elif system == "Darwin":
            subprocess.Popen(["open", DEFAULT_CONFIG_PATH])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", DEFAULT_CONFIG_PATH])
        else:
            typer.launch(DEFAULT_CONFIG_PATH)
    except Exception as e:
        typer.echo(typer.style(f"Failed to open the configuration file: {e}", fg=typer.colors.YELLOW))

@app.command(
    "clear",
    short_help=help_texts["clear"]["short_help"],
    help=auto_format_help(help_texts["clear"]["detailed_help"]),
)
def clear_config(
    section: Optional[str] = typer.Argument(
        None,
    )
):
    ensure_config_exists()
    
    allowed_sections = set(TOP_CONFIG_CLASS_MAP.keys())
    if section:
        if section not in allowed_sections:
            typer.echo(typer.style(f"Invalid section '{section}'. Allowed sections: {', '.join(allowed_sections)}.", fg=typer.colors.YELLOW))
            raise typer.Exit()
        current_config = load_config()
        current_config[section] = generate_default_section_config(section)
        save_config(current_config)
        typer.echo(f"Configuration section '{section}' has been reset to default.")
    else:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            os.remove(DEFAULT_CONFIG_PATH)
            typer.echo("Entire configuration file deleted from disk.")
        else:
            typer.echo("No configuration file found to delete.")

@app.command(
    "list",
    short_help=help_texts["list"]["short_help"],
    help=auto_format_help(help_texts["list"]["detailed_help"]),
)
def list_config(
    section: Optional[str] = typer.Argument(
        None,
    )
):
    ensure_config_exists()
    
    current_config = load_config()
    if section in TOP_CONFIG_CLASS_MAP.keys():
        # Retrieve the specified section and print its contents directly.
        if section not in current_config:
            typer.echo(f"No configuration found for section '{section}'.")
            return
        typer.echo(f"Current configuration for '{section}':")
        typer.echo(yaml.dump(current_config[section], default_flow_style=False, sort_keys=False))
    else:
        typer.echo("Current configuration:")
        for sec in TOP_CONFIG_CLASS_MAP.keys():
            if sec in current_config:
                typer.echo(f"**{sec}**:")
                typer.echo(yaml.dump(current_config[sec], default_flow_style=False, sort_keys=False))

# Define a function to poll for config changes.
def wait_for_config_update(sent_remote_training_key, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        config_data = load_config()  # Your function to load the config file.
        registered_remote_training_key = config_data.get("rllib", {}).get("remote_training_key")
        if sent_remote_training_key == registered_remote_training_key:
            return config_data
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for config update.")

def connect_to_remote_rl_server(region: str, env_config: Dict) -> str:
    
    ws = websocket.WebSocket()
    if region not in ["us-east-1", "us-east-2", "ap-northeast-2"]:
        raise ValueError(f"Invalid region: {region}")
    
    remote_rl_server_url = f"wss://{region}.agent-gpt.ccnets.org"
    
    ws.connect(remote_rl_server_url)
    
    register_request = json.dumps({
        "action": "register", 
        "data": env_config
    })
    ws.send(register_request)
    remote_training_key = ws.recv()
    ws.close()    
    
    return remote_rl_server_url, remote_training_key

@app.command(
    "simulate",
    short_help=help_texts["simulate"]["short_help"],
    help=auto_format_help(help_texts["simulate"]["detailed_help"]),
)
def simulate(
    env_type: Optional[str] = typer.Option(None, "--env-type", help="Environment type: 'gym' or 'unity'"),
    env_id: Optional[str] = typer.Option(None, "--env-id", help="Environment ID to simulate, e.g., 'Walker2d-v5'"),
    num_envs: Optional[int] = typer.Option(None, "--num-envs", help="Number of parallel environments"),
    num_agents: Optional[int] = typer.Option(None, "--num-agents", help="Number of agents to simulate and train (1-8)"),
    region: Optional[str] = typer.Option(None, "--region", help="AWS region for simulation/training"),
    entry_point: Optional[str] = typer.Option(None, "--entry-point", help="Entry point script for the simulation"),
    env_dir: Optional[str] = typer.Option(None, "--env-dir", help="Directory containing the simulation environment files"),
):
    env_type = env_type or typer.prompt("Please provide the environment type ('gym' or 'unity')", default="gym")
    env_id = env_id or typer.prompt("Please provide the environment ID (e.g., 'Walker2d-v5')", default="Walker2d-v5")
    num_agents = num_agents or typer.prompt("Please provide the number of agents", type=int, default=256)
    num_envs = num_envs or typer.prompt("Please provide the number of parallel environments between 1~8", type=int, default=4)
    region = region or typer.prompt("Please provide the AWS region (Supported regions: 'us-east-1', 'us-east-2', 'ap-northeast-2'. External users may use other regional servers as well.)", default="us-east-1")

    typer.echo(f"Environment type: {env_type}")
    typer.echo(f"Environment ID: {env_id}")
    typer.echo(f"Number of agents: {num_agents}")
    typer.echo(f"Number of parallel environments: {num_envs}")
    typer.echo(f"AWS region: {region}")
    typer.echo(f"Entry point script: {entry_point}")
    typer.echo(f"Environment directory: {env_dir}")
    
    configs = load_config()
    configs["sagemaker"]["region"] = region
    save_config(configs)
        
    env_config = {
        "env_id": env_id,
        "num_envs": num_envs,
        "entry_point": entry_point,
        "env_dir": env_dir,
    }

    remote_rl_server_url, remote_training_key = connect_to_remote_rl_server(region, env_config)

    # Initial args as list (command-line-style)
    extra_args = [
        "--remote_training_key", remote_training_key,
        "--remote_rl_server_url", remote_rl_server_url,
        "--env_type", env_type,
        "--env_id", env_id,
        "--num_agents", str(num_agents),
        "--num_envs", str(num_envs),
    ]

    # Dynamically add other args from env_config
    for key, value in env_config.items():
        if value is not None:
            extra_args.extend([f"--{key}", str(value)])

    typer.echo("Starting the simulation in a separate terminal window. Please monitor that window for real-time logs.")
    
    simulation_process = open_simulation_in_screen(extra_args)
    try:
        updated_config = wait_for_config_update(remote_training_key, timeout=10)
        
        remote_training_key = updated_config.get("rllib", {}).get("remote_training_key", {})
        typer.echo("Remote Training Key for simulation updated successfully:")
        
        dislay_output = "**Remote Training Key Under**\n" + yaml.dump(remote_training_key, default_flow_style=False, sort_keys=False)
        typer.echo(typer.style(dislay_output.strip(), fg=typer.colors.GREEN))
    except TimeoutError:
        typer.echo("Configuration update timed out. Terminating simulation process.")
        simulation_process.terminate()
        simulation_process.wait()

def initialize_sagemaker_access(
    role_arn: str,
    region: str,
    service_type: str,  # expected to be "trainer" or "inference"
    email: Optional[str] = None
):
    """
    Initialize SageMaker access by registering your AWS account details.

    - Validates the role ARN format.
    - Extracts your AWS account ID from the role ARN.
    - Sends the account ID, region, and service type to the registration endpoint.
    
    Returns True on success; otherwise, returns False.
    """
    # Validate the role ARN format.
    if not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+$", role_arn):
        typer.echo(typer.style("Invalid role ARN format.", fg=typer.colors.YELLOW))
        return False

    try:
        account_id = role_arn.split(":")[4]
    except IndexError:
        typer.echo("Invalid role ARN. Unable to extract account ID.")
        return False

    typer.echo("Initializing access...")
    
    beta_register_url = "https://agentgpt-beta.ccnets.org"
    payload = {
        "clientAccountId": account_id,
        "region": region,
        "serviceType": service_type
    }
    if email:
        payload["Email"] = email
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(beta_register_url, json=payload, headers=headers)
    except Exception:
        typer.echo("Request error.")
        return False

    if response.status_code != 200:
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

    if response.text.strip() in ("", "null"):
        typer.echo("Initialization succeeded.")
        return True

    try:
        data = response.json()
    except Exception:
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

    if data.get("statusCode") == 200:
        typer.echo("Initialization succeeded.")
        return True
    else:
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

import re

def _validate_sagemaker_role_arn(role_arn):
    """
    Validate SageMaker role ARN.
    Raises ValueError if invalid.
    """
    if not role_arn:
        raise ValueError("Role ARN cannot be empty.")

    arn_regex = r"^arn:aws:iam::\d{12}:role\/[\w+=,.@\-_\/]+$"
    if not re.match(arn_regex, role_arn):
        raise ValueError(f"Invalid SageMaker role ARN: {role_arn}")

@app.command(
    "train",
    short_help=help_texts["train"]["short_help"],
    help=auto_format_help(help_texts["train"]["detailed_help"])
)
def train():
    ensure_config_exists()

    config_data = load_config()

    role_arn = config_data.get("sagemaker", {}).get("role_arn")
    if not role_arn:
        role_arn = typer.prompt("Please enter your IAM role ARN for SageMaker access")
        config_data["sagemaker"]["role_arn"] = role_arn
        save_config(config_data)

    region = config_data.get("sagemaker", {}).get("region")
    if not region:
        region = typer.prompt("Please enter the AWS region for SageMaker access")
        config_data["sagemaker"]["region"] = region
        save_config(config_data)

    # Validate role ARN, retry prompt if invalid
    while True:
        try:
            _validate_sagemaker_role_arn(role_arn)
            break
        except ValueError as e:
            print(e)
            role_arn = typer.prompt("Please enter a valid IAM role ARN for SageMaker access")
            config_data["sagemaker"]["role_arn"] = role_arn
            save_config(config_data)

    # Validate role ARN, retry prompt if invalid
    while True:
        try:
            _validate_sagemaker_role_arn(role_arn)
            break
        except ValueError as e:
            print(e)
            role_arn = typer.prompt("Please enter a valid IAM role ARN for SageMaker access")
            config_data["sagemaker"]["role_arn"] = role_arn
            save_config(config_data)
    
    output_path = config_data.get("sagemaker", {}).get("trainer", {}).get("output_path")
    output_path = typer.prompt("Please enter the S3 output path for SageMaker training jobs(e.g., 's3://remoterl' but ensure that the bucket exists and you have write access to it).", default=output_path)
    config_data["sagemaker"]["trainer"]["output_path"] = output_path
    save_config(config_data)       
    print("region:", region)
    print("role_arn:", role_arn)
    print("output_path:", output_path)
        
    email = typer.prompt("Please enter your email address for registration (leave blank to skip)", default="")
    email = email.strip() or None
    initialize_sagemaker_access(role_arn, region, "trainer", email)
    
    input_config_names = ["sagemaker", "rllib"] 
    input_config = {}
    for name in input_config_names:
        input_config[name] = config_data.get(name, {})
    converted_obj = convert_to_objects(input_config)
    
    sagemaker_obj: SageMakerConfig = converted_obj["sagemaker"]
    rllib_config: RLLibConfig = converted_obj["rllib"]
    
    typer.echo("Submitting training job...")
    estimator = RemoteRL.train(sagemaker_obj, rllib_config)
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.command(
    "infer",
    short_help=help_texts["infer"]["short_help"],
    help=auto_format_help(help_texts["infer"]["detailed_help"])
)
def infer():
    ensure_config_exists()

    config_data = load_config()

    role_arn = config_data.get("sagemaker", {}).get("role_arn")
    if not role_arn:
        role_arn = typer.prompt("Please enter your IAM role ARN for SageMaker access")
        config_data["sagemaker"]["role_arn"] = role_arn
        save_config(config_data)

    region = config_data.get("sagemaker", {}).get("region")
    if not region:
        region = typer.prompt("Please enter the AWS region for SageMaker access")
        config_data["sagemaker"]["region"] = region
        save_config(config_data)

    # Validate role ARN, retry prompt if invalid
    while True:
        try:
            _validate_sagemaker_role_arn(role_arn)
            break
        except ValueError as e:
            print(e)
            role_arn = typer.prompt("Please enter a valid IAM role ARN for SageMaker access")
            config_data["sagemaker"]["role_arn"] = role_arn
            save_config(config_data)

    # Validate role ARN, retry prompt if invalid
    while True:
        try:
            _validate_sagemaker_role_arn(role_arn)
            break
        except ValueError as e:
            print(e)
            role_arn = typer.prompt("Please enter a valid IAM role ARN for SageMaker access")
            config_data["sagemaker"]["role_arn"] = role_arn
            save_config(config_data)
    
    model_data = config_data.get ("sagemaker", {}).get("inference", {}).get("model_data")
    model_data = typer.prompt("Please enter the S3 output path for SageMaker inference(e.g., 's3://remoterl/model.tar.gz').", default=model_data)
    config_data["sagemaker"]["inference"]["model_data"] = model_data
    save_config(config_data)       
    print("region:", region)
    print("role_arn:", role_arn)
    print("model_data:", model_data)
            
    email = typer.prompt("Please enter your email address for registration (leave blank to skip)", default="")
    email = email.strip() or None
    initialize_sagemaker_access(role_arn, region, "inference")
        
    # Use the sagemaker configuration.
    input_config_names = ["sagemaker"]
    input_config = {name: config_data.get(name, {}) for name in input_config_names}
    converted_obj = convert_to_objects(input_config)
    
    sagemaker_obj: SageMakerConfig = converted_obj["sagemaker"]

    typer.echo("Deploying inference endpoint...")
    
    inference_api = RemoteRL.infer(sagemaker_obj)
    
    typer.echo(f"Inference endpoint deployed: {inference_api.endpoint_name}")

@app.callback()
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("No command provided. Displaying help information:\n")
        typer.echo(ctx.get_help())
        raise typer.Exit()

if __name__ == "__main__":
    app()   
