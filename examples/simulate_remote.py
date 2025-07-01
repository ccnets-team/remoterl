"""Launch a ``CartPole-v1`` simulator via RemoteRL.

Run this before executing any of the trainer examples so the learning jobs can
connect to a running environment.
"""

import remoterl
from remoterl.config import ensure_api_key

API_KEY = "your_api_key_here" # Replace with your actual RemoteRL API key
ROLE = "simulator"

def main() -> None:
    """Open a RemoteRL connection and wait for work."""
    try:
        # 1️⃣  Connect to the service with your API key
        remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE) # simulator block at init
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
