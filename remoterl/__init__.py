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
from ._internal.bootstrap import init_remoterl
from ._internal.shutdown import shutdown_remoterl


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
        RemoteRL Cloud API key.  May be *None* when using an on-premise hub.
    role
        Either ``"trainer"`` or ``"simulator"``.
    num_workers
        Worker processes to spawn when acting as a trainer.
    num_env_runners
        Environment-runner tasks per simulator when acting as a trainer.

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
