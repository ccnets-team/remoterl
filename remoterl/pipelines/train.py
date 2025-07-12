#!/usr/bin/env python3
"""train.py - Universal RL training dispatcher for RemoteRL

This module serves as a unified entry point for training with different reinforcement learning 
backends (Gymnasium, Stable-Baselines3, or Ray RLlib) using RemoteRL. It parses command-line 
arguments or configuration inputs to determine which training pipeline to run and then hands off 
execution to the appropriate module (`gym.py`, `sb3.py`, or `rllib.py`). By centralizing the logic 
here, the `remoterl train` CLI command can seamlessly support multiple frameworks with a consistent 
interface.

**Supported frameworks:** The first argument (or config key) selects the RL framework:
- `"gym"` – Use the basic Gymnasium rollout (no learning, random policy) for quick tests.
- `"sb3"` – Use Stable-Baselines3 for training an agent (e.g. PPO, DQN, etc.).
- `"rllib"` – Use Ray RLlib for distributed training algorithms.

**How it works:** When you run `remoterl train <framework> [options]`:
1. The dispatcher resolves your RemoteRL API key (similar to the simulator, via environment or config) 
   and initializes the connection with `remoterl.init(role="trainer", ...)`, honoring any remote-specific 
   settings like `num_workers` if provided.
2. It then imports the corresponding module for the chosen framework and calls its `train_<framework>` 
   function, passing along the parsed hyperparameters.
3. If the required library for that framework is not installed or any configuration is invalid, it will 
   print a clear error message and exit gracefully.

**Usage (CLI):** Run via the `remoterl` CLI. For example:
"""
from __future__ import annotations

import itertools
import importlib
import sys
from typing import Any, Dict, List, Tuple

import typer
import remoterl  # pip install remoterl
from remoterl.config import resolve_api_key
from .helpers import filter_config, safe_print

# ---------------------------------------------------------------------------
# Supported back-ends (alias → (module, train_fn))
# ---------------------------------------------------------------------------
_RL_FRAMEWORK_MAP: Dict[str, Tuple[str, str]] = {
    "gym": ("gym", "train_gym"),
    "gymnasium": ("gym", "train_gym"),
    "ray": ("rllib", "train_rllib"),
    "rllib": ("rllib", "train_rllib"),
    "sb3": ("sb3", "train_sb3"),
    "stable_baselines3": ("sb3", "train_sb3"),
}

_RL_FRAMEWORK_FULL_NAMES: Dict[str, Tuple[str, str]] = {
    "gym": "gymnasium",
    "gymnasium": "gymnasium",
    "ray": "ray[rllib]",
    "rllib": "ray[rllib]",
    "sb3": "stable-baselines3",
    "stable_baselines3": "stable-baselines3",
}

# ---------------------------------------------------------------------------
# CLI-tail → dict helper (kept internal)
# ---------------------------------------------------------------------------

def _parse_cli_tail(tokens: List[str]) -> Dict[str, Any]:
    """Convert a raw token list (as passed by *Typer*'s ``ctx.args``) into a dict."""
    out: Dict[str, Any] = {}
    extra_positional: List[str] = []

    it = iter(tokens)
    first = next(it, None)
    if first is not None:
        if first.startswith("--"):
            # First token is already a flag → rewind
            it = itertools.chain([first], it)
        else:
            out["env_id"] = first
    for tok in it:
        if tok.startswith("--"):
            key = tok.lstrip("-").lower().replace("-", "_")
            val = next(it, True)  # lone switch → True
            if isinstance(val, str) and val.startswith("--"):
                # The flag was a boolean switch; rewind
                out[key] = True
                it = itertools.chain([val], it)
            else:
                out[key] = val
        else:
            extra_positional.append(tok)
    if extra_positional:
        out["extra_positional"] = extra_positional
    return out

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def train(rl_framework: str, params: List[str] | Dict[str, Any] | None = None) -> None:
    """Dispatch to the selected *RL framework* with the parsed configuration."""
    # ------------------------------------------------------------------
    # Parameter normalisation
    # ------------------------------------------------------------------
    if params is None:
        params = {}
    if isinstance(params, list):  # raw CLI tail → parse
        params = _parse_cli_tail(params)
    if not isinstance(params, dict):
        typer.secho("*params* must be a dict or list of tokens.", fg="red", bold=True)
        raise typer.Exit(code=1)

    # lower-case all keys to align with dataclass attributes
    params = {k.lower(): v for k, v in params.items()}


    # ------------------------------------------------------------------
    # API-key + RemoteRL init (same logic as before)
    # ------------------------------------------------------------------
    api_key = resolve_api_key()
    if not api_key:
        sys.exit(
            "No RemoteRL API key found.\n"
            "Set REMOTERL_API_KEY or run `remoterl register` first."
        )
    typer.echo(f"**RemoteRL Training | Framework={_RL_FRAMEWORK_FULL_NAMES[rl_framework]} | key={api_key[:8]}...**")

    # Initialise networking (blocks if remote not reachable)
    params.pop("api_key", None)  # remove api_key from params if present
    params.pop("role", None)  # remove role from params if present
    
    remoterl_kwargs = filter_config(remoterl.init, params).copy()
    
    # RemoteRL directly receives required "num_workers" and "num_env_runners" parameters when using Ray RLlib.
    if rl_framework in {"ray", "rllib"}:
        remoterl_kwargs.pop("num_workers", None)  # remove num_workers if present, as it is not used by Ray RLlib
        remoterl_kwargs.pop("num_env_runners", None)  # remove num_workers if present, as it is not used by Ray RLlib

    is_remote = remoterl.init(api_key, role="trainer", **remoterl_kwargs)      # chainable    
    if not is_remote:
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Dynamic import & training
    # ------------------------------------------------------------------
    module_name, train_fn_name = _RL_FRAMEWORK_MAP[rl_framework]

    try:
        module = importlib.import_module(f".{module_name}", __package__)
        online_train_fn = getattr(module, train_fn_name)
    except ModuleNotFoundError as err:
        typer.secho(str(err), fg="red", bold=True)
        raise typer.Exit(code=1)

    # Run the RL framework-specific training loop
    online_train_fn(params)


# ---------------------------------------------------------------------------
# Stand-alone execution (useful for tests)                                    
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Example usage: python train.py gym --lr 3e-4 CartPole-v1
    if len(sys.argv) >= 2:
        rl_framework_arg = sys.argv[1]
        extra = sys.argv[2:]
        train(rl_framework_arg, extra)
    else:
        safe_print("Usage: train.py <rl_framework> [--flag value ...] [ENV_ID]")
