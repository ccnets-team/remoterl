"""Helper functions for configuration filtering and type handling.

This module provides utility functions to process configuration dictionaries, 
mainly to ensure that only valid parameters are passed to functions or classes 
and to coerce values into expected types. It is used by the training back-ends 
(SB3 and RLlib) to filter user-provided hyperparameters before passing them to 
library functions and to handle optional or union types from function signatures.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Union, get_origin, get_args

# -----------------------------------------------------------------------------
# ─────────────────────────── 1. Utility helpers ──────────────────────────────
# -----------------------------------------------------------------------------
def _canonical(anno):
    """Determine the runtime type to cast to from a type annotation.

    This internal helper extracts the concrete type(s) from an annotation, 
    simplifying generic and optional types:
    - If the annotation is `inspect._empty` (no annotation), return None (no casting).
    - If it's a Union (including Optional), return the underlying type(s) excluding None.
      e.g., Optional[int] -> int, Union[int, str] -> (int, str).
    - Otherwise, return the annotation as is (which should be a type or class).

    Args:
        anno: A type annotation object (from function signature).

    Returns:
        Type or tuple of Types or None: The type(s) to use for casting values. 
        Returns None if there is no specific type to cast to.
    """
    if anno is inspect._empty:
        return None                      # no annotation → leave as-is
    origin = get_origin(anno)
    if origin is Union:                  # Optional[...] or other unions
        args = [a for a in get_args(anno) if a is not type(None)]
        return args[0] if len(args) == 1 else tuple(args)
    return anno

def filter_config(func, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a config dict to parameters accepted by a function, casting types appropriately.

    This utility inspects the signature of the given function (or callable) and 
    returns a new dictionary containing only the entries from `cfg` that correspond 
    to parameters of that function. It also attempts to coerce values to the expected 
    types based on the function's type annotations for basic types (int, float, str, bool).

    The type casting is helpful when configuration comes from sources like CLI args 
    (strings) or YAML, ensuring the values match the expected types of the function parameters.

    Args:
        func: The function or callable whose parameters we want to match.
        cfg (Dict[str, Any]): A dictionary of configuration parameters (possibly containing extra keys).

    Returns:
        Dict[str, Any]: A filtered dictionary with keys that are valid parameters of `func`. 
        Values are cast to the function's annotated types when applicable (for int, float, str, bool).
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

from builtins import print as _print
import sys

def safe_print(*args, **kwargs) -> None:
    """Print to stdout with best-effort Unicode safety."""
    kwargs.setdefault("flush", True)
    text = " ".join(map(str, args))
    try:
        _print(text, **kwargs)
    except UnicodeEncodeError:
        safe = text.encode(sys.stdout.encoding or "ascii", "replace") \
                   .decode(sys.stdout.encoding or "ascii", "replace")
        _print(safe, **kwargs)
    except Exception:
        _print(text, **kwargs)