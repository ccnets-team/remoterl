#!/usr/bin/env python3
"""
Run a RemoteRL simulator.

Lookup order for the API key via :func:`resolve_api_key`:
  1. ``$REMOTERL_API_KEY`` environment variable
  2. ``~/.config/remoterl/config.json`` (or ``REMOTERL_CONFIG_PATH``)
"""
from __future__ import annotations
import sys, typer
import remoterl   # pip install remoterl
from remoterl.config import resolve_api_key
    
def simulate(*args, **kwargs) -> None:
    api_key = resolve_api_key()
    if not api_key:
        sys.exit(
            "No RemoteRL API key found.\n"
            "Set REMOTERL_API_KEY or run `remoterl register` first."
        )
    typer.echo(f"**RemoteRL Simulator Started with API Key:** {api_key[:8]}... (resolved)\n")

    try:
        remoterl.init(api_key, role="simulator")  # blocks until Ctrl-C
    except KeyboardInterrupt:
        # graceful shutdown already handled by remoterl; just exit
        typer.echo("Simulation aborted by user.", err=True)
        raise SystemExit(1)

# input *args, **kwargs to allow for future extensions
if __name__ == "__main__":
    simulate()
