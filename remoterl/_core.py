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

def init(
    api_key: Optional[str],
    role: Literal["trainer", "simulator"],
    *,
    num_workers: int = 1,
    num_env_runners: int = 2,
    max_env_runners: int = 32,
) -> bool:
    """Initialise RemoteRL networking.

    Parameters
    ----------
    api_key
        RemoteRL Cloud API key.  May be *None* when using an on-premises.
    role
        Either ``"trainer"`` or ``"simulator"``.
    num_workers(only trainer parameter)
        Number of local workers to launch.  Defaults to **1**.
    num_env_runners(only trainer parameter)
        Number of remote environment-runner processes to launch.  Defaults to **2**.
    max_env_runners(only simulator parameter)
        Maximum number of environment-runner processes to launch.  Defaults to **32** for simulators

    Raises
    ------
    RuntimeError
        Propagated unchanged from the compiled core when a remote peer is
        unavailable or any other initialisation error occurs.
    NotImplementedError
        Raised when the **RemoteRL package is not available**
        (this source-only stub is being imported instead of the wheel).
    """
    raise NotImplementedError(
        "The RemoteRL package is not available. "
        "Run `pip install remoterl` first."
    )


def shutdown() -> None:
    """Shut down RemoteRL networking.

    Terminates every worker and environment-runner process associated with the
    current device and its active remote session.

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
        Raised when the **RemoteRL package is not available**
        (this source-only stub is being imported instead of the wheel).
    """
    raise NotImplementedError(
        "The RemoteRL package is not available. "
        "Run `pip install remoterl` first."
    )
