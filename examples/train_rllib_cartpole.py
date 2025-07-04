"""Train ``Breakout`` with RLlib over RemoteRL.

This example mirrors the CartPole variant but uses an Atari environment via the
Gymnasium interface.
"""

import ray, remoterl
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from remoterl.config import ensure_api_key

API_KEY = "your_api_key_here"
ROLE    = "trainer"
ENV_ID  = "CartPole-v1" # "ALE/Breakout-v5"  

def main() -> None:
    # ─── 1️⃣  Connect to RemoteRL ────────────────────────────────────────────
    if not remoterl.init(api_key=ensure_api_key(API_KEY), role=ROLE):
        return

    # ─── 2️⃣  Build AlgorithmConfig with version guards ─────────────────────
    try:  # ---------- Modern builder API (RLlib ≥ 2.10) ----------
        algo_config: AlgorithmConfig = (
            get_trainable_cls("PPO").get_default_config()
            .environment(env=ENV_ID)
            .env_runners(num_env_runners=4, num_envs_per_env_runner=16)
            .rollouts(rollout_fragment_length="auto", sample_timeout_s=None)
            .training(num_epochs=20, train_batch_size=1_000,
                      minibatch_size=256, lr=1e-3)
        )
    except AttributeError:  # ---------- Legacy dict API ----------
        algo_config = get_trainable_cls("PPO").get_default_config()
        # Environment & rollout workers
        algo_config["env"] = ENV_ID
        algo_config["num_workers"] = 4
        algo_config["num_envs_per_worker"] = 16
        algo_config["rollout_fragment_length"] = "auto"
        algo_config["sample_timeout_s"] = None
        # Training hyper-params
        algo_config["num_sgd_iter"]        = 20     # ≈ num_epochs
        algo_config["train_batch_size"]    = 1_000
        algo_config["sgd_minibatch_size"]  = 256
        algo_config["lr"]                  = 1e-3

    # ─── 3️⃣  Force the v1 stack when possible ──────────────────────────────
    for flag in ("enable_rl_module_and_learner",
                 "enable_env_runner_and_connector_v2"):
        try:
            setattr(algo_config, flag, False)
        except AttributeError:
            pass  # older RLlib didn’t expose these flags – safe to ignore

    # ─── 4️⃣  Spin up Ray quietly ───────────────────────────────────────────
    ray.init()

    print("Algorithm configuration:", algo_config)

    # ─── 5️⃣  Train! ─────────────────────────────────────────────────────────
    try:
        algo = algo_config.build_algo()
    except AttributeError:  # Legacy API
        algo = get_trainable_cls("PPO")(config=algo_config)
        
    try:
        results = algo.train()
        print("Training completed. Results:", results)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        results = None
    finally:
        ray.shutdown()
        
    return results

if __name__ == "__main__":
    main()