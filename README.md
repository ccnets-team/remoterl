<p align="center">
  <img src="https://github.com/user-attachments/assets/973a48c3-829c-4b4c-9479-48341ba518a4" alt="RemoteRL Banner" style="max-width:100%; height:auto;"/>
</p>

# RemoteRL – Remote Reinforcement Learning for Everyone, Everywhere 🚀
> **Cloud-native RL in a single line of code**

[![PyPI](https://img.shields.io/pypi/v/remoterl)](https://pypi.org/project/remoterl/)
[![Python](https://img.shields.io/pypi/pyversions/remoterl)](https://pypi.org/project/remoterl/)
[![Dependencies](https://img.shields.io/librariesio/release/pypi/remoterl)](https://pypi.org/project/remoterl/)

* **[Installation](#-installation)** · **[Configure Key](#-configure-your-api-key)** · **[Hello‑World Example](#-hello-world-example)** · **[Next Steps](#-next-steps)**

---

## 🧩 How It Works — Three Pieces Mental Model


<div align="center">

Simulator(s)/Robot(s)&emsp;⇄&emsp;🌐 RemoteRL Relay&emsp;⇄&emsp;Trainer (GPU/Laptop)

</div>


<p align="center">
  <img width="723" alt="RemoteRL Service Flow" src="https://github.com/user-attachments/assets/c792c0f1-4461-4b13-ade4-0b344eb9ff69" />
</p>

> The **trainer** sends actions, the **simulator** steps the environment, and the relay moves encrypted messages between them. Nothing else to install, no ports to open.
>
> * **Isolated runtimes** – trainer and simulator can run different Python or OS stacks.
> * **Elastic scale** – fan in 1…N simulators, or fan out distributed learner workers.
> * **Always encrypted, never stored** – payloads travel via TLS and are dropped after delivery.
> * **Free tier:** every account includes **1 GB of data credit(per month)** (≈ 1 M CartPole steps).


---


## 📦 Installation

```bash
# Gymnasium only (lightweight)
pip install remoterl

# + Stable‑Baselines3
pip install remoterl stable-baselines3

# + Ray RLlib
pip install remoterl "ray[rllib]" torch pillow
```

---

## 🔐 Configure Your API Key

To use RemoteRL, you need an API key.
You can get it either via CLI or from the website:

### Option 1 — CLI (Recommended)

```bash
remoterl register  # Opens browser and fetches your API key automatically
```
The key will be saved to your local config automatically.

### Option 2 — Manual (for server, CI, or scripts)
1. Visit [remoterl.com/signup](https://remoterl.com/signup) and **sign up for an account**
2. Go to your Dashboard
3. Copy your API key

Set it as an environment variable:
```bash
export REMOTERL_API_KEY=api_xxxxx...
```


## 💻 Hello World Example

### Run **two terminals**:

```bash
# Terminal A – simulator
$ remoterl simulate

# Terminal B – trainer
$ remoterl train 
```
  
<details>

<summary><code>remoterl simulate</code> — Python example</summary>

```python
import remoterl

# 1. Decide at runtime whether this process is the trainer or the simulator
remoterl.init(role="simulator")  # blocks
remoterl.shutdown()  # optional
```
</details>

<details>
<summary><code>remoterl train</code> — Python example</summary>

```python
import gymnasium as gym
import remoterl

remoterl.init(role="trainer")        # one call switches to remote mode

env = gym.make("CartPole-v1")        # actually runs on the simulator
obs, _ = env.reset()
for _ in range(1_000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```
</details>

That’s it – you’ve split CartPole across the network.


## ⚡ Latency & Isolation

| Path                            | Typical RTT | Notes                               |
| ------------------------------- | ----------- | ----------------------------------- |
| Trainer ↔ same‑region simulator | 10‑50 ms    | Feels like local play.              |
| Trainer ↔ cross‑continent       | 50‑150 ms   | Use frame‑skip for twitchy control. |

---

## 📚 Next Steps


* **[`Cloud Service Overview`](<./docs/Overview/overview-cloud-service.md>)** – details on what is and how it works.  
* **[`Console-Output Guide`](<./docs/SDK (Python)/sdk-console-output-guide.md>)** –  step-by-step screenshots from a *live* trainer ↔ simulator session, with every line called out explained.  
* **[`Quick-Start (Init & Shutdown)`](<./docs/SDK (Python)/sdk-quick-start-init-shutdown.md>)** – step-by-step examples of `remoterl.init()` and `remoterl.shutdown()` for trainers and simulators.  
* **[`Trainer Cheat-sheet`](<./docs/SDK (Python)/sdk-trainer-remote-call-cheat-sheet.md>)** – Gymnasium, Stable-Baselines3, and RLlib one-liners for remote execution.  

## 📎 Quick Links

- 🔑 [Get your API Key](https://remoterl.com) – Create an account on the official site to get your key.
- 📊 [RemoteRL Dashboard](https://remoterl.com/user/dashboard) – Manage your usage, keys, and settings.
- 📘 [Documentation Index](./docs/Overview/overview-cloud-service.md) – Start from the top-level service overview.

---
 
## 📄 License 

RemoteRL is distributed under a commercial licence.
We offer a free tier, while premium plans help offset our worldwide cloud-server costs. See [`LICENSE`](./LICENSE.txt) for details.

---


**Happy remote training!** 🎯
