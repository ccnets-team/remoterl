"""Train ``CartPole-v1`` (or an Atari game) with Ray RLlib (PPO) over RemoteRL.

This script demonstrates using Ray's RLlib library to train a PPO agent on a remote environment. 
By default it runs CartPole-v1, but you can set an Atari game (e.g., Breakout) as the `ENV_ID` 
to test a more complex scenario. Launch one or more simulators first (with the same API key), 
and ensure that Ray RLlib is installed before running this trainer.

**Prerequisites**

* RemoteRL API key – set the `REMOTERL_API_KEY` env var or edit `API_KEY`.  
  Get one at <https://remoterl.com/user/dashboard>.
* Ray RLlib **plus** Torch (needed for PPO) **and** Pillow (image preprocessing).

    # Linux / macOS
    pip install remoterl "ray[rllib]" torch pillow

    # Windows (RLlib ≥ 2.45.0 has issues on Win; pin below)
    pip install remoterl "ray[rllib]<2.45.0" torch pillow

"""
import ray, remoterl
from remoterl.config import ensure_api_key
from remoterl.pipelines.rllib import configure_algorithm

API_KEY = "your_api_key_here"
ROLE    = "trainer"
ENV_ID  = "CartPole-v1" # "ALE/Breakout-v5"  

def main() -> None:
    # ─── 1️⃣  Connect to RemoteRL ────────────────────────────────────────────
    if not remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE):
        return

    # ─── 2️⃣  Build AlgorithmConfig with version guards ─────────────────────
    hyperparameters = {
        "trainable_name": "PPO",
        "env": ENV_ID,
        "num_env_runners": 4,
        "num_envs_per_env_runner": 16,
        "rollout_fragment_length": "auto",
        "sample_timeout_s": None,
        "num_epochs": 20,
        "train_batch_size": 1_000,
        "minibatch_size": 256,
        "lr": 1e-3,
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
    }
    algo_config = configure_algorithm(hyperparameters)

    # ─── 3️⃣  Spin up Ray quietly ───────────────────────────────────────────
    ray.init()  

    print("Algorithm configuration:", algo_config)

    # ─── 4️⃣  Build the Algorithm ─────────────────────────────────────────
    try:
        algo = algo_config.build_algo()
    except AttributeError:  # Legacy API
        raise RuntimeError(
            "This example requires Ray RLlib version 2.10 or later. "
            "Please upgrade your Ray installation."
        )
        
    # ─── 5️⃣  Train the Algorithm ────────────────────────────────────────────
    try:
        results = algo.train()
        print("Training completed. Results:", results)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        results = None
    finally:
        # ─── 6️⃣  Clean up ────────────────────────────────────────────────
        ray.shutdown()
        
    return results

if __name__ == "__main__":
    main()