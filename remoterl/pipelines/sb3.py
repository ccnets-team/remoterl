#!/usr/bin/env python3
"""
sb3_backend.py – Stable-Baselines3 backend with RLlib-style meta structure
============================================================================
Drop-in replacement for the older, smaller sb3_backend.py.

Key points
----------
* Pure **plain-dict API** → `train_sb3(hyperparams: dict)`.
* Internal helpers (`filter_config`, `ensure_default_hyperparams`) match those
  in rllib_backend.py so all back-ends feel the same.
* Works across SB3 releases ≥ 1.6.
* No external `sb3_config.py` needed – everything lives in this file.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Union, get_origin, get_args

import typer

try:
    import stable_baselines3 as sb3
except ModuleNotFoundError as err:  # pragma: no cover
    raise ModuleNotFoundError(
        "Backend 'sb3' selected but *stable-baselines3* is not installed.\n"
        "Install it with:\n\n"
        "    pip install stable-baselines3[extra]\n"
    ) from err

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


def guess_ppo_hyperparams(n_envs, max_episode_steps, target_rollout: int = 2048) -> tuple[int, int]:
    '''Heuristic for (n_steps, n_epochs) based on env length & parallelism.'''
    ep_len = max_episode_steps if max_episode_steps is not None else 1000
    ep_len = int(ep_len)  # ensure it's an int
    n_envs = int(n_envs or 1)  # ensure it's an int
    target_rollout = int(target_rollout)
    if ep_len <= 50:
        n_steps = min(5 * ep_len, 2048)
    elif ep_len >= 3000:
        n_steps = 1024
    else:
        n_steps = max(16, target_rollout // n_envs)

    rollout = n_steps * n_envs
    if rollout >= 4096:
        n_epochs = 3
    elif rollout >= 2048:
        n_epochs = 5
    else:
        n_epochs = 8

    return n_steps, n_epochs

def ensure_default_hyperparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    """SB3 defaults + optional PPO auto-tune."""
    defaults = {
        "env_id": "CartPole-v1",
        "n_envs": 32,
        "algo": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 200_000,
        # ----- PPO specifics -----
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        # ----- Misc -----
        "device": "auto",
        "verbose": 1,
        "log_dir": str(Path.cwd() / "sb3_logs"),
    }

    merged = {**defaults, **(hp or {})}

    # Auto-tune n_steps / n_epochs if user left them blank
    if merged.get("algo", "PPO").upper() == "PPO":
        if merged.get("n_steps") is None or merged.get("n_epochs") is None:
            ep_len_guess = 1000
            n_s, n_e = guess_ppo_hyperparams(merged.get("n_envs", 1), ep_len_guess)
            merged.setdefault("n_steps", n_s)
            merged.setdefault("n_epochs", n_e)

    # Normalisation
    merged["algo"] = merged["algo"].upper()
    merged["n_envs"] = int(merged["n_envs"])
    merged["total_timesteps"] = int(merged["total_timesteps"])

    # Drop None values
    merged = {k: v for k, v in merged.items() if v is not None}
    return merged


# ─────────────────────────── 2. Algo registry ───────────────────────────────
# -----------------------------------------------------------------------------
_ALGOS: Dict[str, Type[sb3.common.base_class.BaseAlgorithm]] = {
    "PPO": sb3.PPO,
    "A2C": sb3.A2C,
    "DQN": sb3.DQN,
    "SAC": sb3.SAC,
    "TD3": sb3.TD3,
    "DDPG": sb3.DDPG,
}

# -----------------------------------------------------------------------------
# ─────────────────────────── 3. Main entry-point ─────────────────────────────
# -----------------------------------------------------------------------------
def train_sb3(hyperparams: Dict[str, Any]) -> None:
    """
    Train an SB3 agent – plain-dict contract shared by all RemoteRL back-ends.

    Parameters
    ----------
    hyperparams : dict
        Any mix of SB3 or convenience keys. Unknown keys are silently ignored.

    Returns
    -------
    None
        The model is saved to disk directly.
    """
    hyperparams = ensure_default_hyperparams((hyperparams or {}).copy())  # never mutate caller

    # ------------------------------------------------------------------
    # 1) Environment construction (vector or single)
    # ------------------------------------------------------------------
    from stable_baselines3.common.env_util import make_vec_env
    vec_env_kwargs = filter_config(make_vec_env, hyperparams)
    env = make_vec_env(**vec_env_kwargs)      # chainable    
    
    # ------------------------------------------------------------------
    # 2) Algorithm selection
    # ------------------------------------------------------------------
    algo_cls = _ALGOS.get(hyperparams.get("algo"), sb3.PPO)
    model_cfg = filter_config(algo_cls, hyperparams)
    policy = model_cfg.pop("policy", "MlpPolicy")  # default policy
    typer.echo(f"[INFO] Using algorithm: {algo_cls.__name__} with policy: {policy} \
        and rest config: {model_cfg}")
    model = algo_cls(policy=policy, env=env, **model_cfg)

    # ------------------------------------------------------------------
    # 3) Logging setup
    # ------------------------------------------------------------------
    from stable_baselines3.common.logger import configure as sb3_configure
    if "log_dir" in hyperparams:
        log_dir = Path(hyperparams["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.set_logger(sb3_configure(str(log_dir), ["stdout", "csv"]))
        except Exception as exc:
            typer.echo(f"[ERROR] Failed to set up logging: {exc}", err=True)
            pass

    # ------------------------------------------------------------------
    # 4) Learning
    # ------------------------------------------------------------------
    learn_cfg = filter_config(model.learn, hyperparams)
    try:
        typer.echo(f"[INFO] learning parameters: {learn_cfg}")
        model.learn(**learn_cfg)
    except Exception as exc:
        typer.echo(f"[ERROR] Learning failed: {exc}", err=True)
        
    # ------------------------------------------------------------------
    # 5) Save and tidy up
    # ------------------------------------------------------------------
    env.close()
    try:
        _algo = algo_cls.__name__.lower()
        _env_id = vec_env_kwargs.get('env_id',"")
        model_path = f"{_algo}_{_env_id}.zip"
        model.save(model_path)
        typer.echo(f"[OK] Model saved to {model_path}")
    except Exception as exc:
        typer.echo(f"[ERROR] Failed to save model: {exc}", err=True)
        
    return 


# ---------------------------------------------------------------------------
# Stand-alone (quick smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    train_sb3({})
