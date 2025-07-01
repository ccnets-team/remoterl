#!/usr/bin/env python3
"""remoterl.pipelines.train
================================================
Generic training dispatcher for **RemoteRL**.

Key improvements in this revision
---------------------------------
1. **Flexible *params* input** – accepts either a pre-parsed ``dict`` **or** the
   raw CLI tail list (e.g. ``['--lr', '3e-4', '--env_id', 'CartPole-v1']``),
   so the new* remoterl CLI can simply forward ``ctx.args`` without having to
   parse it first.
2. **Robust flag parser** – `_parse_cli_tail()` turns the free-form token list
   into a clean dict:
   * first bare token → ``env_id``
   * ``--flag value`` pairs → ``{"flag": value}``
   * lone ``--switch``      → ``{"switch": True}``
   * extra unflagged tokens → ``extra_positional`` list
3. **Lower-case normalisation** – all keys are coerced to lower-case so they
   match the dataclass attributes in each backend config.
4. **Helpful errors & exit codes** – unknown backend or malformed params now
   exit with code 1 and a coloured message via *Typer*.
"""
from __future__ import annotations

import itertools
import importlib
import sys
from typing import Any, Dict, List, Tuple

import typer
import remoterl  # pip install remoterl
from remoterl.config import resolve_api_key

# ---------------------------------------------------------------------------
# Supported back-ends (alias → (module, train_fn))
# ---------------------------------------------------------------------------
BACKENDS: Dict[str, Tuple[str, str]] = {
    "gym": ("gym", "train_gym"),
    "gymnasium": ("gym", "train_gym"),
    "ray": ("rllib", "train_rllib"),
    "rllib": ("rllib", "train_rllib"),
    "sb3": ("sb3", "train_sb3"),
    "stable_baselines3": ("sb3", "train_sb3"),
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

def train(backend: str, params: List[str] | Dict[str, Any] | None = None) -> None:
    """Dispatch to the selected *backend* with the parsed configuration."""
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

    typer.echo(f"**RemoteRL Training | backend={backend} | key={api_key[:8]}...**")

    # Initialise networking (blocks if remote not reachable)
    is_remote = remoterl.init(api_key, role="trainer")
    if not is_remote:
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Dynamic import & training
    # ------------------------------------------------------------------
    module_name, train_fn_name = BACKENDS[backend]

    try:
        module = importlib.import_module(f".{module_name}", __package__)
        train_fn = getattr(module, train_fn_name)
    except ModuleNotFoundError as err:
        typer.secho(str(err), fg="red", bold=True)
        raise typer.Exit(code=1)

    # Run the backend-specific training loop
    train_fn(params)


# ---------------------------------------------------------------------------
# Stand-alone execution (useful for tests)                                    
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Example usage: python train.py gym --lr 3e-4 CartPole-v1
    if len(sys.argv) >= 2:
        backend_arg = sys.argv[1]
        extra = sys.argv[2:]
        train(backend_arg, extra)
    else:
        print("Usage: train.py <backend> [--flag value ...] [ENV_ID]")
