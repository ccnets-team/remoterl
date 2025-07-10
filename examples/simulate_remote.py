"""Launch a ``CartPole-v1`` simulator via RemoteRL.

Use this script to host the CartPole environment as a **remote simulator**.  
Run it **before** starting any trainer example (e.g., `train_gym_cartpole.py`,  
`train_sb3_cartpole.py`) so those jobs can connect to a live environment.

**Requirements**

* A RemoteRL account and API key – replace the `API_KEY` placeholder in the
  script *or* set the `REMOTERL_API_KEY` environment variable.
  
  Get one at <https://remoterl.com/user/dashboard>.
  
    pip install remoterl
  
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
