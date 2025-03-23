import os
import yaml
import time
from typing import List, Dict

from remoterl import __version__ as CURRENT_REMOTE_RL_VERSION

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.remoterl/config.yaml")
TOP_CONFIG_CLASS_KEYS = ["rllib", "sagemaker"]

def load_config() -> Dict:
    config = {}    
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config

def save_config(config_data: Dict) -> None:
    config_data["version"] = CURRENT_REMOTE_RL_VERSION
    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

def wait_for_config_update(sent_remote_training_key, timeout=10):
    start_time = time.time()
    from .config import load_config
    while time.time() - start_time < timeout:
        config_data = load_config()  # Your function to load the config file.
        registered_remote_training_key = config_data.get("rllib", {}).get("remote_training_key")
        if sent_remote_training_key == registered_remote_training_key:
            return config_data
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for config update.")

def get_nested_config(config, *keys, default=None):
    for key in keys:
        config = config.get(key, {})
    return config or default
    
from ..remote_config import RemoteConfig
def generate_default_section_config(section: str) -> Dict:
    return RemoteConfig().to_dict().get(section)

def generate_default_config() -> Dict:
    return RemoteConfig().to_dict()

def ensure_config_exists():
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        new_config = generate_default_config()
        save_config(new_config)
    else:
        config = load_config()
        if not config or (config.get("version") != CURRENT_REMOTE_RL_VERSION):
            new_config = generate_default_config()
            save_config(new_config)

def convert_to_objects(config_data: Dict) -> Dict:
    """
    Instantiate top-level configuration objects and apply stored config_data.
    """
    new_config = RemoteConfig()
    new_config.set_config(**config_data)
    return new_config

def parse_value(value: str):
    """
    Try converting the string to int, float, or bool.
    If all conversions fail, return the string.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    if value is not None:
        lower = value.lower()
        if lower in ["true", "false"]:
            return lower == "true"
    return value

def parse_extra_args(args: List[str]) -> Dict:
    """
    Parses extra CLI arguments provided in the form:
      --key value [value ...]
    Supports nested keys via dot notation, e.g.:
      --env_host.local1.env_endpoint "http://example.com:8500"
    Returns a nested dictionary of the parsed values.
    """
    new_changes = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # remove the leading "--"
            i += 1
            # Gather all subsequent arguments that do not start with '--'
            values = []
            while i < len(args) and not args[i].startswith("--"):
                values.append(args[i])
                i += 1

            # Determine if we have no values, a single value, or multiple values.
            if not values:
                parsed_value = None
            elif len(values) == 1:
                parsed_value = parse_value(values[0])
            else:
                parsed_value = [parse_value(val) for val in values]

            # Build a nested dictionary using dot notation.
            keys = key.split(".")
            d = new_changes
            for sub_key in keys[:-1]:
                d = d.setdefault(sub_key, {})
            d[keys[-1]] = parsed_value
        else:
            i += 1
    return new_changes

def recursive_update(target, changes: Dict, prefix="") -> List:
    """
    Recursively update attributes of an object (or dictionary) using a nested changes dict.
    Only updates existing attributes/keys.

    Returns:
        list of tuples: Each tuple is (full_key, old_value, new_value, updated, message),
        where 'updated' is a boolean indicating if an update was performed.
    """
    update_log = []
    if isinstance(target, dict):
        for k, new_val in changes.items():
            if k in target:
                current_key = f"{prefix}.{k}" if prefix else k
                if isinstance(new_val, dict):
                    update_log.extend(recursive_update(target[k], new_val, prefix=current_key))
                else:
                    if target[k] != new_val:
                        old_val = target[k]
                        target[k] = new_val
                        update_log.append((current_key, old_val, new_val, True, ""))
                    else:
                        update_log.append((current_key, target[k], new_val, False, "value unchanged"))
            # Do not add new keys.
    else:
        for attr, new_val in changes.items():
            if not hasattr(target, attr):
                continue
            current_val = getattr(target, attr)
            current_key = f"{prefix}.{attr}" if prefix else attr
            if isinstance(new_val, dict):
                update_log.extend(recursive_update(current_val, new_val, prefix=current_key))
            else:
                if current_val != new_val:
                    old_val = current_val
                    setattr(target, attr, new_val)
                    update_log.append((current_key, old_val, new_val, True, ""))
                else:
                    update_log.append((current_key, current_val, new_val, False, "value unchanged"))
    return update_log

def update_config_by_dot_notation(config_obj, new_changes) -> List:
    """
    Applies changes from a nested dict to the configuration objects.

    Returns:
        list of tuples: Each tuple is (key, old_value, new_value, changed, message)
    """
    update_log = []
    for key, new_val in new_changes.items():
        # Allow shorthand syntax for top-level config sections.
        if key in config_obj and isinstance(new_val, dict) and len(new_val) == 1:
            inner_key, inner_value = list(new_val.items())[0]
            key = inner_key
            new_val = inner_value

        matching_obj = None
        for obj in config_obj.values():
            if hasattr(obj, key):
                matching_obj = obj
                break
        
        if matching_obj is None:
            update_log.append((key, None, None, False, "it was not found in the configuration"))
            continue
        
        attr = getattr(matching_obj, key)
        if callable(attr):
            if not isinstance(new_val, list):
                new_val = [new_val]
            converted_args = [parse_value(arg) for arg in new_val if arg is not None]
            try:
                if converted_args:
                    attr(*converted_args)
                else:
                    attr()
            except Exception as e:
                error_msg = f"Error calling '{key}' with arguments {converted_args}: {e}"
                # print(error_msg)
                update_log.append((key, None, None, False, error_msg))
                continue
            arg_str = " ".join(str(x) for x in converted_args)
            update_log.append((key, None, None, True, arg_str))
        elif isinstance(new_val, dict):
            update_log.extend(recursive_update(attr, new_val, prefix=key))
        else:
            current_val = getattr(matching_obj, key)
            if current_val != new_val:
                setattr(matching_obj, key, new_val)
                update_log.append((key, current_val, new_val, True, ""))
            else:
                update_log.append((key, current_val, new_val, False, "value unchanged"))
    return update_log

def update_config_using_method(args: List[str], config_obj: Dict) -> List:
    """
    Process method calls in the config command.

    For example, if the user runs:
      remoterl config simulator set simim --hosting local

    Returns a List of tuples (target_key, keyword_args, placeholder, changed, message).
    """
    if len(args) < 3:
        return [("error", None, None, False, "Not enough arguments for method call")]

    object_name = args[0]  # e.g., "simulator"
    behaviour = args[1]    # e.g., "set" or "del"
    identifier = args[2]   # e.g., "simim"
    method_name = f"{behaviour}_{object_name.lower().replace('-', '_')}"
    method_args_raw = args[3:]
    keyword_args = parse_extra_args(method_args_raw) if method_args_raw else {}

    target_obj = None
    for key, obj in config_obj.items():
        if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
            target_obj = obj
            break

    if target_obj is None:
        message = f"No configuration object found for '{object_name}' using '{behaviour}' for identifier '{identifier}'"
        return [(f"{behaviour} {object_name}", None, None, False, message)]
    try:
        method = getattr(target_obj, method_name)
    except Exception as e:
        return [(f"{behaviour} {object_name}", keyword_args, None, False, str(e))]

    try:
        method(identifier, **keyword_args)
        message = f"{behaviour.capitalize()} '{object_name}' for identifier '{identifier}'" if keyword_args else f"for identifier '{identifier}'"
        return [(f"{behaviour} {object_name}", None, None, True, message)]
    except Exception as e:
        return [(f"{behaviour} {object_name}", None, None, False, str(e))]