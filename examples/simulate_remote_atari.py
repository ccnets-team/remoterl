"""Start an Atari simulator using ALE-py for RemoteRL.

Run this before launching any trainer so the remote workers have an Atari
environment to interact with.

This example uses the `ale-py` package to create Atari environments compatible with
https://ale.farama.org/gymnasium-interface/

pip install "gymnasium[atari, accept-rom-license]"

"""
# pip install ale-py
import gymnasium as gym
import ale_py
import remoterl
from remoterl.config import ensure_api_key

API_KEY = "your_api_key_here"  # Replace or set REMOTERL_API_KEY
ROLE = "simulator"

def main() -> None:
    # ALE-py documentation recommends registering Atari environments via
    gym.register_envs(ale_py)

    # 1️⃣  Connect as a RemoteRL simulator
    try:
        remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE) # simulator block at init
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
