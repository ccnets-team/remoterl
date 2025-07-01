# RemoteRL – Example Scripts

This repository contains **concise examples** that demonstrate how to launch simulators and trainers with the RemoteRL client SDK.

---

## 1 · Installation

```bash
# Gymnasium only (light‑weight)
pip install remoterl

# + Stable‑Baselines3 extras
pip install "remoterl[stable-baselines3]"

# + Ray RLlib extras
pip install "remoterl[rllib]"
```

---

## 2 · Configure your API key

Authenticate in one of two ways:

1. **Interactive CLI (recommended)**

   ```bash
   remoterl register
   ```

   A browser window opens – sign up with Google or email, then copy the API key from your dashboard.

2. **Environment variable**
   Ideal for CI or ad‑hoc sessions:

   ```bash
   export REMOTERL_API_KEY=api_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
   ```

---

## 3 · Running the examples

### 3.1 · CLI‑only workflow

```bash
# Start a local simulator (runs until Ctrl‑C)
remoterl simulate

# In a new terminal – train with the backend of your choice
remoterl train gym                    # Gymnasium
remoterl train sb3   --algo PPO       # Stable‑Baselines3
remoterl train rllib --env CartPole-v1 --num-env-runners 8
```

Additional flags are forwarded directly to the underlying backend. Run:

```bash
remoterl train <backend> --help
```

for the full list.

### 3.2 · Running the raw Python scripts

```bash
python simulate_remote.py             # simulator
python train_gym_cartpole.py          # trainer – Gymnasium
python train_sb3_cartpole.py          # trainer – Stable‑Baselines3
python train_rllib_cartpole.py        # trainer – RLlib
```

---

Happy remote training! 🚀
