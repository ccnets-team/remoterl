"""Train `CartPole-v1` using the simple Gymnasium pipeline.

This script shows the absolute basics of how to connect to the RemoteRL
service and drive a single environment with a random policy.  Launch the
simulator first (with the same API key) and then run this trainer using the same API key.
You will see the remote environment in action. (Requires Gymnasium to be installed.)

**Prerequisites**

* RemoteRL API key – set the `REMOTERL_API_KEY` env var or edit `API_KEY`.  
  Get one at <https://remoterl.com/user/dashboard>.

    pip install remoterl
    
"""
    
import time
import typer
import gymnasium
import remoterl
from remoterl.config import ensure_api_key

API_KEY = "your_api_key_here" # Replace with your actual RemoteRL API key
ROLE = "trainer" 
ENV_ID = "CartPole-v1"

def main() -> None:
    """
    Minimal trainer demo for CartPole-v1 over RemoteRL.

    Steps:
    1.  Connect to the RemoteRL service (role="trainer").
    2.  Create a remote Gymnasium environment just like you would locally.
    3.  Run N episodes with a random policy and print the total reward.
    """

    # 1️⃣  Connect – replace API_KEY with your own or set REMOTERL_API_KEY
    if not remoterl.init(api_key=ensure_api_key(API_KEY), role="trainer"):
        return                      # exit if the backend / simulators are unreachable

    # 2️⃣  Make the (remote) environment.  Nothing special here!
    env = gymnasium.make(ENV_ID)

    # 3️⃣  Roll out a few episodes with a random policy
    NUM_EPISODES = 5
    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done, total_reward, frames = False, 0.0, 0

        while not done:
            action             = env.action_space.sample()    # random action
            obs, reward, term, trunc, _ = env.step(action)
            done               = term or trunc
            total_reward      += float(reward)
            frames            += 1

        print(f"Episode {ep}: return={total_reward:.1f}  frames={frames}")

    env.close()  # always close the env when you’re done

if __name__ == "__main__":
    main()
