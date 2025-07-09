# RemoteRL: Cloud Service for Remote Reinforcement Learning

## Overview

**RemoteRL** is a cloud-based platform that enables reinforcement-learning (RL) training to run remotely, decoupling the agent’s training process from environment (simulator) execution. It’s a scalable framework that lets you **connect environments (simulators or robots) from anywhere**, stream live experience data, and train smarter models in real time. You always run both pieces yourself—the trainer on a laptop, on-prem cluster, or robot controller, and the simulators or robots wherever they already live, nearby or across the globe. RemoteRL’s relay nodes stream data between them over secure WebSockets, letting you connect environments anywhere, send live experience data, and plug into popular RL frameworks—all without rewriting code.

**Key Characteristics:**

* **Separation of Concerns:** The **RL environment (“Simulator”)** keeps running wherever it already lives—on your PC, a robot, or another server—and connects to RemoteRL’s relay. The **learning algorithm (“Trainer”)** stays on your own machine or cluster, receiving observations and rewards through RemoteRL and sending back actions or updated policies. Because each process remains on hardware you control, you never have to install the simulator on the training host, simplifying setup and letting you leverage local compute when it makes sense.

* **Use Cases:** RemoteRL is especially useful for scenarios like **robotics and IoT** (where a physical robot or device provides real-time data to train a model remotely) and **distributed simulation** (where many simulators run in parallel on different machines to speed up training). It allows training "from anywhere" – whether the environment is on-premises or in the cloud – by handling the networking, synchronization, and scaling behind the scenes. For example, a robotics engineer could connect a robot’s control loop to RemoteRL and have a cloud-based RL agent train on the live data, or a researcher could run dozens of game simulations on local PCs all feeding into a single cloud RL learner.

## Core Features and Functionality

RemoteRL provides a number of features to make remote and distributed RL **easy to integrate and manage**:

* **Zero-Setup Integration:** It works out-of-the-box with popular Python RL libraries. Users simply install the `remoterl` Python package and add a one-line initialization with their API key to an existing Gymnasium or RLlib script – *“Run `pip install remoterl` and add `remoterl.init(api_key="…")` to any Gymnasium / Ray RLlib script, and hit run — no rewrites, zero friction”*. This means **no extensive code refactoring or complex cluster setup** is required; RemoteRL hooks into the RL frameworks to handle remote environment communication transparently.

* **Framework and Tool Support:** The service **auto-integrates with popular RL frameworks** like **OpenAI Gymnasium (Gym)** for environment interfaces and **Ray RLlib** for distributed training algorithms. It also supports integration with libraries like **Stable-Baselines3**. This broad support allows developers to use familiar APIs and algorithms (e.g. RLlib’s trainers or Stable-Baselines3’s agents) while RemoteRL manages the remote execution of environments. Whether you use custom Gym environments or standard ones, or whether you train with RLlib’s scalable architecture or a single-agent approach, RemoteRL can plug in with minimal changes.

* **Live Web Dashboard & Monitoring:** RemoteRL provides a **web‑based dashboard** that shows which simulators and trainers are online alongside basic byte/step counters. When you add `remoterl.init(api_key="…")` to a process, it appears on the dashboard within seconds. The dashboard displays **connection status only**—RemoteRL does **not** view, store, or inspect observation, reward, or model payloads. Operators can disconnect a simulator or temporarily block new sessions; they cannot peek into in‑flight data nor pause individual training iterations.
  
* **Distributed & Scalable Training:** RemoteRL’s relay layer is **logically centralized**, but it feeds data just as well into a **single-process trainer** or a **distributed RL cluster** you run with frameworks such as Ray RLlib. Many simulators—or whole fleets of robots—can stream experience to one learner, while an RLlib-style trainer can shard its own work behind the relay. RemoteRL handles fan-in/fan-out, step ordering, and back-pressure, so you scale from one to hundreds of simulators without touching networking code or changing your training loop.

* **Real-Time Online RL:** RemoteRL streams observations and rewards to the trainer—and actions back to simulators—in real time. Each simulator automatically connects through the relay node in the nearest cloud region, so every `env.step()` returns with minimal round-trip delay even when trainer and environments are on different continents. With servers rolling out across major cities, you can run a trainer in Europe while simulators in Asia and the Americas each route to their closest node, delivering interactive, online RL at global scale.

* **Usage-Based Pricing and Tiers:** RemoteRL offers a **free tier** and a premium plan. The service will be *“available free for everyone”* for light use, meaning you can run training jobs without paying as long as data usage is under a certain limit. The **free tier includes 1 GB of data traffic per month** (as noted in the feature list) and likely some reasonable number of simulators. For heavier usage, **premium services (pay-as-you-go)** are offered – *“charged per GB”* for data beyond the free quota. The premium tier is designed for those who need **“extensive data usage, multiple simulators, and large-scale training across global regions.”** In other words, large experiments or enterprise deployments with many environments and high throughput would incur a usage-based fee. This pricing model is **traffic/duration-based** (billing on both bytes of data exchanged and active connection minutes), ensuring you only pay for what you use, and there is **no hard limit on speed or number of simulators in premium** aside from practical scaling limits. This makes the service accessible to individual researchers (on the free tier) while scaling up to industrial projects on the paid tier.

## Architecture and How It Works

At a high level, RemoteRL follows a **client–server architecture specialized for RL**. The **core components** are defined as follows:

* **Trainer:** The RL algorithm process that you run and manage (e.g., on your own cloud instances, on-prem servers, or a laptop). RemoteRL simply provides the networking bridge: it relays state/reward data from simulators to your trainer and returns actions. You can still scale the trainer across multiple machines—using Ray RLlib or similar frameworks—while RemoteRL abstracts away the connection details, not the computation itself.

* **Simulator:** The environment process that produces observations and receives actions—whether it’s a game engine, robotics simulator, custom code, or a real-world device such as a robot control loop or IoT sensor. It can run anywhere (local machine, edge device, or another cloud) and connects to RemoteRL over a secure WebSocket. Functionally it behaves like any Gym environment; the only difference is that these interactions travel across the network through RemoteRL’s bridge.

```python
import gymnasium as gym
import remoterl 

remoterl.init(api_key="YOUR_KEY", role="trainer") # the SDK treats this process as the trainer
env = gym.make("Humanoid-v5")  # Environment runs remotely in another city
```

**Communication:** When you integrate RemoteRL, the typical RL loop is split between the trainer and simulator via the network:

* The **trainer invokes** an environment step (e.g. calling `env.step(action)` in RL code). Instead of running locally, this call is routed through RemoteRL’s library which sends the action to the remote simulator over the WebSocket connection.
* The **simulator receives** the action, applies it (advancing the simulation or executing on the robot), and returns the resulting observation, reward, and done flag back over the socket.
* The trainer then uses the received data to update the agent’s model (e.g., via policy gradient, Q-learning update, etc.), and the cycle repeats.
