import os
import inspect
from typing import Any, Dict, Union, get_origin, get_args
import typer

try:
    # Plain import makes sure Ray itself is present
    import ray  # noqa: F401
    # Pull in one canonical RLlib symbol to verify the sub-package
    from ray.rllib.algorithms import AlgorithmConfig  # noqa: F401
    from ray.tune.registry import get_trainable_cls    

except (ModuleNotFoundError, ImportError) as err:  # pragma: no cover
    raise ModuleNotFoundError(
        "Backend 'rllib' selected but Ray RLlib is not installed.\n"
        "Install it with:\n\n"
        "    pip install 'remterl[rllib]' or \n"
        "    pip install 'ray[rllib]==2.42.0' \n"
    ) from err
    
def _canonical(anno):
    """
    Return the concrete runtime type we should cast to
    (e.g. Optional[int] -> int, Union[int, str] -> (int, str)).
    """
    if anno is inspect._empty:
        return None                      # no annotation → leave as-is
    origin = get_origin(anno)
    if origin is Union:                  # Optional[...] or other unions
        args = [a for a in get_args(anno) if a is not type(None)]
        return args[0] if len(args) == 1 else tuple(args)
    return anno

def filter_config(func, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sub-set *cfg* to parameters accepted by *func* **and**
    try to co-erce basic types (int, float, bool, str) to match annotations.
    """
    sig = inspect.signature(func)
    out = {}
    for k, v in cfg.items():
        if k not in sig.parameters or k == "self":
            continue

        tgt = _canonical(sig.parameters[k].annotation)
        if tgt in (int, float, str):
            try:
                v = tgt(v)
            except Exception:
                pass                       # keep original; let SB3 complain
        elif tgt is bool:
            if isinstance(v, str):
                if v.lower() in {"true", "1", "yes", "y"}:
                    v = True
                elif v.lower() in {"false", "0", "no", "n"}:
                    v = False

        out[k] = v
    return out

def ensure_default_hyperparams(hyperparams: dict) -> dict:
    """Fill in any missing RLlib hyper-parameters with sensible defaults."""
    # ➊ Pick the environment value in priority order.
    env_val = (
        hyperparams.get("env")                # modern key
        or hyperparams.pop("env_id", None)    # legacy alias (and remove it)
        or "CartPole-v1"                      # hard default
    )

    # ➋ All other defaults.
    defaults = {
        "env":                     env_val,
        "num_env_runners":         4,
        "num_envs_per_env_runner": 16,
        "num_epochs":              20,
        "train_batch_size":        1_000,
        "minibatch_size":          256,
        "lr":                      1e-3,
        "num_learners":            1,
        "num_gpus_per_learner":    1,
        "rollout_fragment_length": "auto",
        "sample_timeout_s":        None,
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
    }

    # ➌ Merge: user-provided keys win, defaults fill the gaps.
    return {**defaults, **hyperparams}

def configure_algorithm(hyperparams: Dict[str, Any]) -> Union[AlgorithmConfig, Dict[str, Any]]:
    """
    Return an RLlib `AlgorithmConfig` (builder API) when the current RLlib
    build supports it, otherwise return a legacy dict config.  All requested
    sub-sections are applied only when they exist.
    """
    # ------------------------------------------------------------------ #
    # ❶  Static overrides required by RemoteRL
    # ------------------------------------------------------------------ #

    trainable_name: str = hyperparams.pop("trainable_name", "PPO")
    hyperparams = ensure_default_hyperparams(hyperparams)

    # ------------------------------------------------------------------ #
    # ❷  Verify we can obtain the trainable class and its default config
    # ------------------------------------------------------------------ #
    try:
        trainable_cls = get_trainable_cls(trainable_name)
    except Exception as err:
        raise RuntimeError(f"RLlib does not recognise trainable '{trainable_name}'.") from err

    get_def_cfg = getattr(trainable_cls, "get_default_config", None)
    if not callable(get_def_cfg):
        raise RuntimeError(
            f"Your RLlib version ({trainable_cls.__module__}) does not expose "
            f"`get_default_config()` for {trainable_name}. Please upgrade RLlib."
        )

    # ------------------------------------------------------------------ #
    # ❸  Obtain the skeleton config object (builder) or dict (very old)
    # ------------------------------------------------------------------ #
    algo_config = get_def_cfg()          # type: Union[AlgorithmConfig, Dict[str, Any]]

    # ------------------------------------------------------------------ #
    # ❹  Sections to populate when their builder methods exist
    # ------------------------------------------------------------------ #
    sub_configs = [
        "resources", "framework", "api_stack", "environment",
        "env_runners", "learners", "training", "evaluation",
        "callbacks", "offline_data", "multi_agent", "reporting",
        "checkpointing", "debugging", "fault_tolerance",
        "rl_module", "experimental",
    ]

    # ------------------------------------------------------------------ #
    # ❺  Apply each section guarded by `callable(method)`
    # ------------------------------------------------------------------ #
    if isinstance(algo_config, AlgorithmConfig):          # modern builder API
        for section in sub_configs:
            # ① Bound method: used for actual builder chaining
            bound_method = getattr(algo_config, section, None)

            # ② Unbound (class-level) method: used for parameter filtering
            unbound_method = getattr(AlgorithmConfig, section, None)

            if callable(bound_method):
                # Safely filter using the class-level signature
                kwargs = filter_config(unbound_method, hyperparams)
                # Actual call is via the instance method
                algo_config = bound_method(**kwargs)      # chainable        
    else:                                                 # legacy dict fall-back
        # Directly merge any leftover hyper-params (safest thing to do)
        algo_config.update(hyperparams)

    return algo_config


def print_cluster_resources():
    # Get available cluster resources from Ray
    resources = ray.cluster_resources()
    ray_gpu_count = resources.get("GPU", 0)
    ray_cpu_count = resources.get("CPU", 0)

    # Read environment variables
    num_gpus = int(os.environ.get("SM_NUM_GPUS", ray_gpu_count))
    num_cpus = int(os.environ.get("SM_NUM_CPUS", ray_cpu_count))

    typer.echo(f"GPU Count: {num_gpus}")
    typer.echo(f"CPU Count: {num_cpus}")

def ensure_ray_init_args(opts: dict) -> dict:
    """
    Return only those key/value pairs that the current Ray build
    actually accepts, so `ray.init(**ensure_ray_init_args(...))` never
    raises an unexpected AttributeError.
    """
    allowed = inspect.signature(ray.init).parameters
    return {k: v for k, v in opts.items() if k in allowed}
            
def train_rllib(hyperparams: Dict[str, Any]):
    """
    Run a single RLlib training loop (Ray already inside SageMaker container).

    The function is forward- and backward-compatible with RLlib versions that
    either return an `AlgorithmConfig` (builder API) or a legacy dict config.
    """
    # ────────────────────────────────────────────────────────────────────
    # 1️⃣  Bring up Ray quietly (same as before)
    # ────────────────────────────────────────────────────────────────────

    # Finally, initialise Ray with the filtered kwargs
    ray.init(**ensure_ray_init_args({
        "runtime_env": {"working_dir": os.path.dirname(os.path.abspath(__file__))},
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }, ))
    
    print_cluster_resources()

    # -------------------------------------------------------------------
    # 2️⃣  Capture trainable_name *before* it gets popped in config_algorithm
    # -------------------------------------------------------------------
    trainable_name = hyperparams.get("trainable_name", "PPO")

    # -------------------------------------------------------------------
    # 3️⃣  Build the AlgorithmConfig (or dict) with all the guards you added
    # -------------------------------------------------------------------
    algo_config = configure_algorithm(hyperparams)
    cfg = algo_config if isinstance(algo_config, dict) else algo_config.to_dict()
    typer.echo(f"Algorithm configuration: {cfg}")


    # -------------------------------------------------------------------
    # 4️⃣  Instantiate the trainer in a version-safe way
    # -------------------------------------------------------------------
    try:
        # Modern builder API (AlgorithmConfig has .build_algo)
        algo = algo_config.build_algo()  # type: ignore[attr-defined]
    except AttributeError:
        # Legacy dict path
        algo = get_trainable_cls(trainable_name)(config=algo_config)

    # -------------------------------------------------------------------
    # 5️⃣  Train, handle any runtime errors gracefully
    # -------------------------------------------------------------------
    try:
        results = algo.train()
        typer.echo(f"Training completed. Results: {results}")
    except Exception as err:
        typer.echo(f"[train_rllib] Training failed: {err}")
        results = None
    finally:
        ray.shutdown()

    return results
