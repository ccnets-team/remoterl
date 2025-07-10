"""_rllib_wrapper.py - Ray RLlib patching utilities for RemoteRL integration.

This module provides functions to monkey-patch certain RLlib behaviors so that 
RemoteRL's remote environments can integrate seamlessly. These patches are applied 
on the trainer side to intercept environment registration and algorithm construction:
- Registering remote environments in RLlib's global registry using a secret token.
- Patching `tune.register_env` to automatically wrap environment creators with RemoteRL logic.
- Patching RLlib's `AlgorithmConfig.build_algo` method to inject a RemoteRL hook before trainer instantiation.

All functions are safe to call only if Ray RLlib is installed; they will raise 
ImportError otherwise. The original functions or methods are returned so they 
can be restored if needed (useful for unit tests or cleanup).
"""

from __future__ import annotations
from typing import Any

try:
    from ray import tune
    _rllib_installed: bool = True
except ImportError:
    _rllib_installed = False
# Ray is optional – only import lazily to avoid hard dependency when users
# run code paths that do not require RLlib.

def patch_rllib_register(env_id: str, remoterl_token: Any) -> None:
    """Register a RemoteRL env with RLlib's global registry.

    env_id :
        The string id of the environment to register. 
    remoterl_token :
        RemoteRL issues an opaque token that represents the current 
        session and its environment runners. Because this token never 
        leaves the local SDK, your real API credentials stay completely 
        private.
    
    Raises
    ------
    ImportError
        If the caller tries to use this helper without Ray installed.
    """
    if not _rllib_installed:  # Fail early with a friendly hint.
        raise ImportError(
            "Ray is not installed. `pip install 'ray[rllib]'`"
        )

    from remoterl._internal.learner._mod0 import remoterl_env_creator
    from ray.tune.registry import _global_registry, ENV_CREATOR
    creator = lambda ctx: remoterl_env_creator(ctx, remoterl_token=remoterl_token)

    _global_registry.register(ENV_CREATOR, env_id, creator)


def patch_rllib_register_env(remoterl_token: Any):
    """Monkey‑patch ``tune.register_env`` so every call is auto‑wrapped.
    The original function is returned so tests can restore default behaviour.
    """
    if not _rllib_installed:
        raise ImportError(
            "Ray is not installed. `pip install 'ray[rllib]'"
        )

    # Keep a handle to the original in case the caller wants to undo.
    _orig = tune.register_env

    from remoterl._internal.learner._mod0 import remoterl_register_env
    def wrapper(env_name, env_creator):
        # Delegate to compiled helper, injecting the token transparently.
        return remoterl_register_env(   
            env_name,
            env_creator,
            remoterl_token=remoterl_token,
        )

    # Perform the monkey‑patch.
    tune.register_env = wrapper

    # Return the original so user can call `orig()` to restore.
    return _orig


def patch_rllib_build_algo(remoterl_token: Any):
    """
    Wrap RLlib’s AlgorithmConfig.build_algo so we can run a RemoteRL
    pre-hook before the real method executes.
    """
    if not _rllib_installed:
        raise ImportError(
            "Ray is not installed. `pip install 'ray[rllib]'"
        )
            
    from ray.rllib.algorithms import algorithm_config
    _orig = algorithm_config.AlgorithmConfig.build_algo  # keep original

    from remoterl._internal.learner._mod0 import remoterl_build_algo         # our hook

    def wrapper(self, *args, **kwargs):
        remoterl_build_algo(self, remoterl_token=remoterl_token)  # inject
        return _orig(self, *args, **kwargs)                       # delegate

    algorithm_config.AlgorithmConfig.build_algo = wrapper         # patch
    
    return _orig  # return original so caller can restore it        
