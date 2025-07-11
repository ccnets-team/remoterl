"""Train ``CartPole-v1`` with Stable Baselines3 (PPO) over RemoteRL.

This example uses the Stable Baselines3 library to train a basic PPO agent 
on the CartPole environment via RemoteRL. Launch your simulator(s) first 
(with the same API key), then run this script to begin training the agent. 
Make sure that `stable-baselines3` is installed on your trainer side.


**Prerequisites**

* RemoteRL API key – set the `REMOTERL_API_KEY` env var or edit `API_KEY`.  
  Get one at <https://remoterl.com/user/dashboard>.

    pip install stable-baselines3

"""
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

import remoterl
from remoterl.config import ensure_api_key

API_KEY = "your_api_key_here"  # Replace with your actual API key
ROLE = "trainer"
ENV_ID = "CartPole-v1"

def main() -> None:
    # 1️⃣  Connect – replace API_KEY with your own or set REMOTERL_API_KEY
    if not remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE):
        return

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = make_vec_env(ENV_ID, n_envs=32)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------
    model = PPO(                # 3️⃣  Instantiate PPO
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
        device="auto",
        verbose=1,
        n_steps=64,
        n_epochs=4,
        batch_size=64,
    )
        
    model.learn(                 # 4️⃣  Train the agent
        total_timesteps=10_000, 
    )

    # ------------------------------------------------------------------
    # Save & clean-up
    # ------------------------------------------------------------------
    env.close()                  # 5️⃣  Close the environment

    print(f"[OK] Training finished\n")


if __name__ == "__main__":
    main()
