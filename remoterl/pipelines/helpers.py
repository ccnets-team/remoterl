from __future__ import annotations

import inspect
from typing import Any, Dict, Union, get_origin, get_args

# -----------------------------------------------------------------------------
# ─────────────────────────── 1. Utility helpers ──────────────────────────────
# -----------------------------------------------------------------------------
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
