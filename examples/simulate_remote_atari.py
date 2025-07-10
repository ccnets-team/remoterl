"""Start an Atari game simulator using ALE-py for RemoteRL.

Use this script to host an Atari environment (via the Arcade Learning Environment) as a remote simulator. 
Run it **before** launching any trainer that will use an Atari game, so the remote agent has an environment 
to interact with. 
This example uses the `ale-py` package to register Atari environments in Gymnasium 
(see the ALE Gymnasium interface documentation).
https://ale.farama.org/gymnasium-interface/

Make sure to install Gymnasium with Atari support and accept the ROM license, for example:

**Prerequisites**

* RemoteRL API key – set the `REMOTERL_API_KEY` env var or edit `API_KEY`.  
  Get one at <https://remoterl.com/user/dashboard>.
  
    pip install remoterl "gymnasium[atari, accept-rom-license]"
    pip install ale-py
    
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
