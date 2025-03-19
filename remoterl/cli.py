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
from .config.rllib import RemoteRLlibConfig
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
    allowed_sections = set(TOP_CONFIG_CLASS_MAP.keys())
    if section:
        ensure_config_exists()
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
                if sec == "rllib":
                    typer.echo("RLlib: (Showing only modified configuration)")
                elif sec == "sagemaker":
                    typer.echo("**SageMaker**: ")
                else:
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

@app.command(
    "simulate",
    short_help=help_texts["simulate"]["short_help"],
    help=auto_format_help(help_texts["simulate"]["detailed_help"]),
)
def simulate(
    env_type: Optional[str] = typer.Option(None, "--env-type", help="Environment type: 'gym' or 'unity'"),
    env: Optional[str] = typer.Option(None, "--env", help="Environment name to simulate, e.g., 'Walker2d-v5'"),
    num_env_runners: Optional[int] = typer.Option(None, "--num-env-runners", help="Number of parallel environments"),
    num_envs_per_env_runner: Optional[int] = typer.Option(None, "--num-envs-per-env-runner", help="Number of envs per worker to simulate and train (1-8)"),
    region: Optional[str] = typer.Option(None, "--region", help="AWS region for simulation/training"),
):
    ensure_config_exists()
    
    configs = load_config()
    
    user_region = configs.get("sagemaker", {}).get("region") 
    if SageMakerConfig.DEFAULT_REGION == user_region:
        user_region = "ap-northeast-2"
    
    default_region = region or user_region   
    
    env_type = env_type or typer.prompt("Please provide the environment type ('gym' or 'unity')", default="gym")
    env = env or typer.prompt("Please provide the environment name (e.g., 'Walker2d-v5')", default="Walker2d-v5")
    num_envs_per_env_runner = num_envs_per_env_runner or typer.prompt("Please provide the number of agents", type=int, default=64)
    num_env_runners = num_env_runners or typer.prompt("Please provide the number of parallel environments between 1~8", type=int, default=4)
    region = region or typer.prompt(
        "Please specify the AWS region. Currently, our service is built for the 'us-east-1' and 'ap-northeast-2' regions; however, external users are welcome to use this server as well.",
        default=default_region
    )
    
    typer.echo(f"Environment type: {env_type}")
    typer.echo(f"Environment ID: {env}")
    typer.echo(f"Number of envs per worker: {num_envs_per_env_runner}")
    typer.echo(f"Number of parallel environments: {num_env_runners}")
    typer.echo(f"AWS region: {region}")

    configs["sagemaker"]["region"] = region
    save_config(configs)
        
    env_config = {
        "env_id": env,
        "num_envs": num_env_runners,
        "entry_point": None,
        "env_dir": None,
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
        typer.secho("Simulation is now running. Please run 'remoterl train' to continue..", fg="green")
    except TimeoutError:
        typer.echo("Configuration update timed out. Terminating simulation process.")
        simulation_process.terminate()
        simulation_process.wait()

def register_beta_access(
    role_arn: str,
    region: str,
    email: Optional[str] = None
) -> bool:
    """
    Register your AWS account details for beta access to the service.

    - Validates the role ARN format.
    - Extracts your AWS account ID from the role ARN.
    - Sends the account ID, region, and service type to the beta registration endpoint.

    Returns True on successful registration; otherwise, returns False.
    """
    try:
        # Validate the role ARN format.
        if not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+$", role_arn):
            raise ValueError("Invalid role ARN format.")

        # Extract the account ID from the role ARN.
        parts = role_arn.split(":")
        if len(parts) < 5 or not parts[4]:
            raise ValueError("Invalid role ARN. Unable to extract account ID.")
        account_id = parts[4]

        typer.echo("Registering beta access...")

        beta_register_url = "https://agentgpt-beta.ccnets.org"
        payload = {
            "clientAccountId": account_id,
            "region": region,
            "serviceType": "remoterl"
        }
        if email:
            payload["Email"] = email

        headers = {'Content-Type': 'application/json'}

        # Send the registration request.
        response = requests.post(beta_register_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError("Registration failed.")

        typer.echo("Registration succeeded.")
        return True

    except Exception as e:
        typer.echo(typer.style(str(e), fg=typer.colors.YELLOW))
        return False

def _validate_sagemaker_role_arn(role_arn: str) -> None:
    """
    Validate SageMaker role ARN.
    Raises ValueError if invalid.
    """
    if not role_arn:
        raise ValueError("Role ARN cannot be empty.")

    arn_regex = r"^arn:aws:iam::\d{12}:role\/[\w+=,.@\-_\/]+$"
    if not re.match(arn_regex, role_arn):
        raise ValueError(f"Invalid SageMaker role ARN: {role_arn}")

def _ensure_s3_output_path(output_path: str) -> str:
    """
    Ensure the provided output path starts with 's3://'.
    Strips leading/trailing whitespace and any trailing slashes,
    and prepends 's3://' if necessary.
    Raises a ValueError if the input is empty.
    """
    output_path = output_path.strip()
    if not output_path:
        raise ValueError("Output path cannot be empty.")
    
    output_path = output_path.rstrip('/')
    
    if not output_path.startswith("s3://"):
        return "s3://" + output_path.lstrip('/')
    
    return output_path

@app.command(
    "train",
    short_help=help_texts["train"]["short_help"],
    help=auto_format_help(help_texts["train"]["detailed_help"])
)
def train():
    ensure_config_exists()

    config_data = load_config()
    
    config_data.setdefault("rllib", {})
    remote_training_key = config_data["rllib"].get("remote_training_key")
    region = config_data["sagemaker"].get("region")
    if not remote_training_key:
        typer.echo("Error: Remote training key not found. Please run 'remoterl simulate' to generate the key.")
        raise typer.Exit()
    else:
        typer.echo(f"Remote training key found: {remote_training_key}")
    while SageMakerConfig.DEFAULT_REGION == region:
        typer.echo("Error: Region information is missing. Please run 'remoterl simulate' to get the region.")
        raise typer.Exit()
    typer.echo(f"Region: {region}")

    # Ensure the 'sagemaker' key exists in the configuration.
    config_data.setdefault("sagemaker", {})
    # Retrieve role ARN from config; prompt if missing.
    role_arn = config_data["sagemaker"].get("role_arn")
    asked_role_arn = False
    if SageMakerConfig.DEFAULT_ROLE_ARN == role_arn:
        role_arn = typer.prompt(
            "Please enter your IAM role ARN for SageMaker access (e.g., "
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole). "
            "You can find your role ARN in the AWS IAM console under Roles."
        )        
        config_data["sagemaker"]["role_arn"] = role_arn
        save_config(config_data)
        asked_role_arn = True
    # Validate role ARN and re-prompt until valid.
    while True:
        try:
            _validate_sagemaker_role_arn(role_arn)
            break
        except ValueError as e:
            typer.echo(f"Error: {e}")
            asked_role_arn = True
            role_arn = typer.prompt("Please enter a valid IAM role ARN for SageMaker access")
            config_data["sagemaker"]["role_arn"] = role_arn
            save_config(config_data)
    
    typer.echo(f"Role ARN: {role_arn}")
        
    # Retrieve the default output path from config, or provide a fallback.
    default_output_path = config_data["sagemaker"].get("output_path")
    while True:
        output_path = typer.prompt(
            "Please enter the S3 output path in your region",
            default=default_output_path
        )
        if output_path == SageMakerConfig.DEFAULT_OUTPUT_PATH:
            typer.echo("Warning: Default output path detected. Please enter a valid S3 path.")
            continue
        try:
            s3_output_path = _ensure_s3_output_path(output_path)
            config_data["sagemaker"]["output_path"] = s3_output_path
            save_config(config_data)            
            break  # Valid input; exit loop.
        except Exception as e:
            typer.echo(f"Error: {e}\nPlease try again.")
    
    # Print the final values for confirmation.
    typer.echo(f"Output Path: {s3_output_path}")
        
    email = typer.prompt("Please enter your email address for registration (leave blank to skip)", default="")
    email = email.strip() or None
    
    register_beta_access(role_arn, region, email)
    
    input_config = {}
    for name in ["sagemaker"] :
        input_config[name] = config_data.get(name, {})
    converted_obj = convert_to_objects(input_config)
    sagemaker_obj: SageMakerConfig = converted_obj["sagemaker"]
    hyperparameters = config_data["rllib"]
    
    typer.echo("Submitting training job...")
    
    estimator = RemoteRL.train(sagemaker_obj, hyperparameters)
    
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.callback()
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("No command provided. Displaying help information:\n")
        typer.echo(ctx.get_help())
        raise typer.Exit()

if __name__ == "__main__":
    app()   
