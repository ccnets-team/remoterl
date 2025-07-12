"""rllib.py - Version-free Ray RLlib integration via typed-config CLI parsing.

This module integrates Ray RLlib into the RemoteRL training workflow. It provides 
the `train_rllib` function, which configures and runs an RLlib training loop 
(using one iteration of training by default) based on a plain dictionary of 
hyperparameters. The code is designed to handle differences between RLlib versions 
(by using the modern `AlgorithmConfig` builder API when available, or falling back 
to older dict-based configs) so that users can write version-agnostic training code.

Key responsibilities:
- Fill in default hyperparameters for RLlib and ensure the environment is specified.
- Construct an RLlib `AlgorithmConfig` or legacy config dict with all relevant 
  sub-config sections applied (resources, environment, training, etc.).
- Initialize Ray and any required RLlib patches, then instantiate the trainer 
  (algorithm) in a version-safe way and run training.
- Provide helper functions for common tasks like filtering Ray init args and 
  retrieving trainable classes by name, to smooth over version and API differences.

This module assumes Ray (with RLlib) is installed. If RLlib is not available, it 
will raise an error with instructions to install the appropriate package.
"""
import os
import inspect
from typing import Any, Dict, Union
import typer

try:
    # Attempt to import Ray and RLlib to verify they are available.
    import ray  # noqa: F401 (just to ensure Ray can be imported)
    from ray.rllib.algorithms import AlgorithmConfig  # noqa: F401 (checks RLlib subpackage)
    from ray.tune.registry import get_trainable_cls, _global_registry    
except (ModuleNotFoundError, ImportError) as err:  # pragma: no cover
    # If Ray or RLlib is not installed or fails to import, raise an error with instructions.
    raise ModuleNotFoundError(
        "RL Framework 'rllib' selected but Ray RLlib is not installed.\n"
        "Install it with:\n\n"
        "    pip install 'ray[rllib]' or \n"
        "    pip install 'ray[rllib]' torch pillow (if you need PyTorch and PPO)\n"
        "    (Note: Ray RLlib versions after 2.44.0 may have issues on Windows.)\n"
    ) from err

from .helpers import filter_config

def ensure_default_hyperparams(hyperparams: dict) -> dict:
    """Ensure essential RLlib hyperparameters are present, using sensible defaults.

    This function populates a given hyperparams dict with default values for 
    critical RLlib settings if they are missing. It guarantees that an environment 
    is specified (using 'env' or legacy 'env_id'), and sets defaults for common 
    training parameters like number of workers, batch sizes, learning rate, etc. 
    These defaults are chosen to be reasonable for a simple training setup.

    Args:
        hyperparams (dict): User-supplied hyperparameters (possibly incomplete).

    Returns:
        dict: A new dictionary containing the original hyperparams augmented with 
        default values for any missing keys.
    """
    # Determine the environment ID in priority order: modern 'env' key, then legacy 'env_id', or default to CartPole.
    env_val = (
        hyperparams.pop("env", None)                # current RLlib config key for env
        or hyperparams.pop("env_id", None)    # accept 'env_id' for backward compatibility (remove it if present)
        or "CartPole-v1"                      # default environment if none provided
    )

    # Define default values for other RLlib config fields.
    defaults = {
        "env":                     env_val,
        "num_env_runners":         2,
        "num_envs_per_env_runner": 16,
        "num_epochs":              20,
        "train_batch_size":        1_000,
        "minibatch_size":          256,
        "lr":                      1e-3,
        "rollout_fragment_length": "auto",
        "sample_timeout_s":        None,
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
    }

    # Merge user hyperparams with defaults (user values take precedence).
    return {**defaults, **hyperparams}

def configure_algorithm(hyperparams: Dict[str, Any]) -> Union[AlgorithmConfig, Dict[str, Any]]:
    """Build an RLlib AlgorithmConfig (or config dict) from given hyperparameters.

    This function creates a full RLlib algorithm configuration object using the 
    provided hyperparameters. If the installed RLlib version supports the builder 
    API (AlgorithmConfig), it will produce an AlgorithmConfig instance; otherwise, 
    it returns a legacy dictionary config. The function applies sub-configurations 
    (resources, environment, training, etc.) if those sections exist in the builder API.

    It handles:
    1. Static overrides required by RemoteRL (ensuring defaults via ensure_default_hyperparams).
    2. Verification that the specified algorithm (trainable) exists and has a default config.
    3. Building the base config object (AlgorithmConfig or dict).
    4. Iterating through known sub-config sections and applying any provided hyperparams to them.
    5. Merging any remaining hyperparams into the config for completeness.

    Args:
        hyperparams (Dict[str, Any]): Hyperparameters including at least 'trainable_name' 
            (the RLlib algorithm to use, e.g., "PPO"). May also include keys corresponding 
            to nested config sections (like 'env', 'train_batch_size', etc.).

    Returns:
        AlgorithmConfig or dict: A fully constructed RLlib configuration. If using RLlib 
        >= 2.x with AlgorithmConfig builder API, this will be an AlgorithmConfig object 
        (which can later be built into an Algorithm). If using an older RLlib, this will 
        be a dictionary of configuration settings.

    Raises:
        RuntimeError: If the requested trainable algorithm name is not recognized by RLlib, 
            or if RLlib's API does not provide a default config for that algorithm (indicating 
            an incompatible RLlib version).
    """
    # 1️⃣ Static overrides and defaults required by RemoteRL.
    trainable_name: str = hyperparams.pop("trainable_name", "PPO")  # Algorithm name to train (default "PPO").
    hyperparams = ensure_default_hyperparams(hyperparams)           # Apply default hyperparameters.

    # 2️⃣ Verify the trainable class is available and provides a default config.
    try:
        trainable_cls = get_trainable_cls(trainable_name)
    except Exception as err:
        # If RLlib cannot find the specified algorithm, raise a clear error.
        raise RuntimeError(f"RLlib does not recognize trainable '{trainable_name}'.") from err

    # Check that the trainable class has a get_default_config method (required for building config).
    get_def_cfg = getattr(trainable_cls, "get_default_config", None)
    if not callable(get_def_cfg):
        raise RuntimeError(
            f"Your RLlib version ({trainable_cls.__module__}) does not expose "
            f"`get_default_config()` for {trainable_name}. Please upgrade RLlib."
        )

    # 3️⃣ Obtain the base configuration object (could be AlgorithmConfig or a dict, depending on RLlib version).
    algo_config = get_def_cfg()  # This yields an AlgorithmConfig instance on newer RLlib, or a dict on older versions.

    # 4️⃣ List of configuration sub-sections to apply if the builder pattern is available.
    sub_configs = [
        "resources", "framework", "api_stack", "environment",
        "env_runners", "learners", "training", "evaluation",
        "callbacks", "offline_data", "multi_agent", "reporting",
        "checkpointing", "debugging", "fault_tolerance",
        "rl_module", "experimental",
    ]

    # 5️⃣ Apply each relevant section in the hyperparams to the config, if the corresponding builder method exists.
    if isinstance(algo_config, AlgorithmConfig):
        # Modern RLlib (builder API): chain configuration calls for each section.
        for section in sub_configs:
            # Bound method of the AlgorithmConfig instance (if it exists).
            bound_method = getattr(algo_config, section, None)
            # Unbound method from AlgorithmConfig class (to inspect its signature for filtering).
            unbound_method = getattr(AlgorithmConfig, section, None)
            if callable(bound_method) and callable(unbound_method):
                # Filter hyperparams to only those valid for this section's method.
                kwargs = filter_config(unbound_method, hyperparams)
                # Apply the section settings by calling the bound method with filtered args.
                algo_config = bound_method(**kwargs)  # chainable config update
    else:
        # Legacy RLlib (dict config): directly merge any remaining hyperparams into the config dictionary.
        algo_config.update(hyperparams)

    return algo_config

def smart_get_trainable(name: str):
    """Find a trainable class in RLlib by trying multiple name variations.

    RLlib algorithms (trainables) might be registered under different names or 
    casing (e.g., "PPO", "PPOTrainer") and custom ones might be registered at runtime. 
    This helper attempts a series of likely name variants as well as any names 
    currently in RLlib's registry to locate the trainable class.

    Args:
        name (str): The base name of the trainable (e.g., "PPO").

    Returns:
        (trainable_cls, str): A tuple of the found trainable class and the exact name 
        variant that was successful. For example, it might return (PPOTrainer, "PPO") 
        or (PPOTrainer, "PPOTrainer").

    Raises:
        RuntimeError: If no matching trainable class is found after trying all variants.
    """
    variants = [name, name.upper(), name.lower(), name.capitalize()]  # ① Try straightforward case modifications.
    # ② Include all trainable names already registered in the current RLlib registry (for completeness).
    variants += list(_global_registry._to_trainable.keys())  # For RLlib ≥ 2.7; older versions might use a different attribute.

    tried = set()
    for cand in variants:
        if cand in tried:
            continue
        tried.add(cand)
        try:
            return get_trainable_cls(cand), cand  # Return the first match (class and name).
        except Exception:
            continue

    # If none of the variants matched a registered trainable, raise an error with all tried names listed.
    raise RuntimeError(
        f"RLlib does not recognize trainable '{name}'. Tried: {', '.join(sorted(tried))}"
    )

def print_cluster_resources():
    """Print the available CPU and GPU resources in the current Ray cluster.

    This uses Ray's resource reporting to fetch the total number of CPUs and GPUs 
    available, taking into account special environment variables (e.g., SageMaker 
    may set SM_NUM_GPUS/CPUS). It then logs these counts via Typer for the user 
    to see (helpful for verifying that Ray sees the correct resources).
    """
    resources = ray.cluster_resources()  # Get cluster-wide resources from Ray.
    ray_gpu_count = resources.get("GPU", 0)
    ray_cpu_count = resources.get("CPU", 0)

    # Check if environment (like SageMaker) specifies resource counts, otherwise use Ray's numbers.
    num_gpus = int(os.environ.get("SM_NUM_GPUS", ray_gpu_count))
    num_cpus = int(os.environ.get("SM_NUM_CPUS", ray_cpu_count))

    typer.echo(f"GPU Count: {num_gpus}")
    typer.echo(f"CPU Count: {num_cpus}")

def ensure_ray_init_args(opts: dict) -> dict:
    """Filter Ray initialization options to only those supported by the current Ray version.

    Ray's `ray.init()` function may not accept all possible arguments in every version. 
    This utility inspects the signature of `ray.init` and removes any keys from the 
    provided options dict that are not present in that signature. This prevents 
    `ray.init(**opts)` from raising an unexpected AttributeError due to an unsupported 
    argument.

    Args:
        opts (dict): A dictionary of potential arguments for `ray.init()`. For example, 
            this might include keys like "address", "ignore_reinit_error", etc.

    Returns:
        dict: A copy of `opts` containing only the keys that `ray.init` can accept.
    """
    allowed_params = inspect.signature(ray.init).parameters  # Get valid parameters for ray.init.
    return {k: v for k, v in opts.items() if k in allowed_params}

def train_rllib(hyperparams: Dict[str, Any]):
    """Train an RLlib algorithm using the provided hyperparameters dictionary.

    This function brings up a Ray cluster (if not already running), configures an RLlib 
    algorithm (trainer) according to the given hyperparameters, and executes one 
    training iteration. It automatically handles differences between RLlib versions: 
    using the new `AlgorithmConfig` builder pattern if available, or falling back to the 
    older config dictionary approach. After training, it shuts down Ray to free resources.

    Args:
        hyperparams (Dict[str, Any]): Hyperparameters and settings for training. This may 
            include Ray initialization options (e.g., 'ignore_reinit_error': True), the 
            RLlib algorithm name under 'trainable_name', environment settings, and any 
            algorithm-specific configs (like 'env', 'lr', 'train_batch_size', etc.).

    Returns:
        Optional[dict]: The results from the training run (for example, RLlib returns a 
        dictionary of training metrics after an iteration). If training fails, returns None.

    Raises:
        RuntimeError: Propagated if configuration fails (e.g., unknown trainable name or 
            incompatible RLlib version).
        ModuleNotFoundError: If Ray RLlib is not installed (raised at import time).
    """
    # ────────────────────────────────────────────────────────────────────
    # 1️⃣  Initialize Ray quietly (with any filtered initialization args)
    # ────────────────────────────────────────────────────────────────────
    ray.init(**ensure_ray_init_args(hyperparams))
    print_cluster_resources()  # Log how many resources (CPUs/GPUs) Ray sees, for user info.

    # -------------------------------------------------------------------
    # 2️⃣  Capture the algorithm name before it gets consumed by configure_algorithm
    # -------------------------------------------------------------------
    trainable_name = hyperparams.get("trainable_name", "PPO")

    # -------------------------------------------------------------------
    # 3️⃣  Build the RLlib AlgorithmConfig or config dict
    # -------------------------------------------------------------------
    algo_config = configure_algorithm(hyperparams)  # Prepare the algorithm configuration.
    cfg = algo_config if isinstance(algo_config, dict) else algo_config.to_dict()
    typer.echo(f"Algorithm configuration: {cfg}")

    # -------------------------------------------------------------------
    # 4️⃣  Instantiate the trainer in a version-safe way
    # -------------------------------------------------------------------
    try:
        # If using modern RLlib (AlgorithmConfig with builder API) – build the Algorithm instance.
        algo = algo_config.build_algo()  # type: ignore[attr-defined]
    except AttributeError:
        # Older RLlib: AlgorithmConfig has no build_algo(). Use the registry to get the trainable class.
        trainable_cls, rotated_name = smart_get_trainable(trainable_name)
        # If AlgorithmConfig instance exists, convert it to dict; if it's already dict, use as is.
        raw_cfg = algo_config if isinstance(algo_config, dict) else algo_config.to_dict()
        # Instantiate the trainer using the resolved class and configuration.
        algo = trainable_cls(config=raw_cfg)
        trainable_name = rotated_name  # Update trainable_name to the variant that was actually used (for logging).

    # -------------------------------------------------------------------
    # 5️⃣  Run training, and handle any runtime errors gracefully
    # -------------------------------------------------------------------
    try:
        results = algo.train()  # Execute one training iteration (returns a results dict).
        typer.echo(f"Training completed. Results: {results}")
    except Exception as err:
        typer.echo(f"[train_rllib] Training failed: {err}")
        results = None
    finally:
        # Always shut down the Ray instance to clean up resources after training.
        ray.shutdown()

    return results
