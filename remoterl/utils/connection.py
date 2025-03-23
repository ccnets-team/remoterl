
from typing import Optional, Dict
import requests
import typer
import re
import json
import websocket

def get_remote_rl_server_url(region: str) -> str:
    if region not in ["us-east-1", "ap-northeast-2"]:
        raise ValueError(f"Invalid region: {region}")
    remote_rl_server_url = f"wss://{region}.ccnets.org"
    return remote_rl_server_url

def connect_to_remote_rl_server(region: str, env_config: Dict) -> str:

    remote_rl_server_url = get_remote_rl_server_url(region)
    
    ws = websocket.WebSocket()
    ws.connect(remote_rl_server_url)
    
    register_request = json.dumps({
        "action": "register", 
        "data": env_config
    })
    ws.send(register_request)
    remote_training_key = ws.recv()
    ws.close()      
    
    return remote_rl_server_url, remote_training_key

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

def validate_sagemaker_role_arn(role_arn: str) -> None:
    """
    Validate SageMaker role ARN.
    Raises ValueError if invalid.
    """
    if not role_arn:
        raise ValueError("Role ARN cannot be empty.")

    arn_regex = r"^arn:aws:iam::\d{12}:role\/[\w+=,.@\-_\/]+$"
    if not re.match(arn_regex, role_arn):
        raise ValueError(f"Invalid SageMaker role ARN: {role_arn}")

def ensure_s3_output_path(output_path: str) -> str:
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