"""Public entry points for :mod:`remoterl`.

Only :func:`init` is a part of the supported API.  They
delegate to the compiled implementation in :mod:`remoterl._internal` and
propagate any :class:`RuntimeError` unchanged so callers may handle it or let it
terminate the program.

Everything under :mod:`remoterl._internal` is private and subject to change.
Use the CLI or the training pipelines instead of calling those modules
directly.
"""

# remoterl/__init__.py
from typing import Optional, Literal
from remoterl import init as init_remoterl, shutdown as shutdown_remoterl

def init(
    api_key: Optional[str],
    role: Literal["trainer", "simulator"],
    *,
    num_workers: int = 1,
    num_env_runners: int = 2,
) -> bool:
    """Initialise RemoteRL networking.

    Parameters
    ----------
    api_key
        RemoteRL Cloud API key.  May be *None* when using an on-premises.
    role
        Either ``"trainer"`` or ``"simulator"``.
    num_workers[only trainer parameter]
        Number of local workers to launch.  Defaults to **1**.
    num_env_runners[only trainer parameter]
        Number of remote environment-runner processes to launch.  Defaults to **2**.

    Raises
    ------
    RuntimeError
        Propagated unchanged from the compiled core when a remote peer is
        unavailable or any other initialisation error occurs.
    """

    return init_remoterl(
        api_key=api_key,
        role=role,
        num_workers=num_workers,
        num_env_runners=num_env_runners,
    )

def shutdown() -> None:
    """Shut down RemoteRL networking.

    Terminates every worker and environment-runner process associated with the
    current device and its active remote session.

    Returns
    -------
    None
    """
    shutdown_remoterl()

__all__ = ["init", "shutdown"]
