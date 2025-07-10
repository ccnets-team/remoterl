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
    # 1️⃣  Connect to the RemoteRL service
    is_connected = remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE)
    if not is_connected:
        return

    # 2️⃣  Create the remote Gymnasium environment
    env = gymnasium.make(ENV_ID)

    """Random-policy rollout for a *single* Gymnasium environment."""
    episode_returns = []
    global_steps = 0
    last_report_time = time.time()
    steps_at_last_report = 0

    for ep in range(10):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_frames = 0
        ep_start = time.time()

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_frames += 1
            global_steps += 1

            now = time.time()
            if now - last_report_time >= 10:
                steps_since = global_steps - steps_at_last_report
                fps = steps_since / (now - last_report_time + 1e-8)
                typer.echo(f"[FPS] {fps:6.1f}")
                last_report_time = now
                steps_at_last_report = global_steps

        episode_returns.append(ep_return)
        typer.echo(
            f"Episode {ep:03d} | return {ep_return:8.2f} | "
            f"frames {ep_frames:5d} | elapsed {time.time() - ep_start:5.2f}s"
        )

    return episode_returns

if __name__ == "__main__":
    main()
