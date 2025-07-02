<p align="center">
  <img src="https://github.com/user-attachments/assets/973a48c3-829c-4b4c-9479-48341ba518a4" alt="RemoteRL Banner" style="max-width:100%; height:auto;"/>
</p>

# RemoteRL – Remote Reinforcement Learning Made Easy 🚀


RemoteRL is a lightweight client SDK for remotely running simulators and reinforcement learning (RL) training jobs.  
It works with **Gymnasium**, **Stable-Baselines3**, and **Ray RLlib** (experimental) backends.  
You can use the CLI or raw Python scripts to launch your remote RL workflows with minimal configuration.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c4249d58-7548-46e4-9888-c99b8260998b" alt="Frame 1080" width="1000"/>
</p>

---




## 📦 Installation

Install with your preferred backend:

**Gymnasium only (lightweight):**
```bash
pip install remoterl
```

**With Stable-Baselines3 support:**
```bash
pip install "remoterl[stable-baselines3]"
```

**With Ray RLlib support (experimental):**
```bash
pip install "remoterl[rllib]" torch
```

---

## 🔐 Configure Your API Key

Before you begin, set your RemoteRL API key. You can do this in two ways:

### 1. Interactive CLI (Recommended)
```bash
remoterl register
```
A browser window will open. Sign in using Google or email, then copy your API key from your RemoteRL dashboard.

### 2. Environment Variable (ideal for CI or quick runs)
```bash
export REMOTERL_API_KEY=api_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
```

---

## 🚀 Quickstart

### Option A · CLI-only Workflow

Start the simulator:
```bash
remoterl simulate
```

In a new terminal, start training using one of the supported backends:

- **Gymnasium**
  ```bash
  remoterl train gym
  ```

- **Stable-Baselines3**
  ```bash
  remoterl train sb3 --algo PPO
  ```

- **Ray RLlib (experimental)**
  ```bash
  remoterl train rllib --env CartPole-v1 --num-env-runners 8
  ```

👉 All arguments are passed through to the backend. Run:
```bash
remoterl train <backend> --help
```
for full options.

---

### Option B · Python Scripts

Prefer to launch via Python scripts? Use the built-in examples:

- **Simulator**
  ```bash
  python simulate_remote.py
  ```

- **Trainer (Gymnasium)**
  ```bash
  python train_gym_cartpole.py
  ```

- **Trainer (Stable-Baselines3)**
  ```bash
  python train_sb3_cartpole.py
  ```

- **Trainer (Ray RLlib, experimental)**
  ```bash
  python train_rllib_cartpole.py
  ```

---

## 🧠 Why RemoteRL?

- 🚀 **Launch RL training remotely** in seconds  
- 🧩 **Modular CLI and SDK** — works with your preferred backend  
- ⚡ **No infrastructure setup required** — connect via API key and go  
- 🧪 **Simple examples** to help you get started fast

---

## 📄 License

This project is licensed under the [COMMERCIAL LICENSE AGREEMENT](LICENSE).

---

**Happy remote training!** 🎯

