# RemoteRL: Cloud Service for Remote Reinforcement Learning

## Overview

**RemoteRL** is a cloud-based platform that enables reinforcement learning (RL) training to be run remotely, decoupling the agent’s training process from the environment (or simulator) execution. It is a scalable framework for remote reinforcement learning that lets users **connect environments (simulators or robots) from anywhere**, stream live experience data, and train smarter models in real time. The service provides a **cloud “trainer” component** which communicates with one or more **remote “simulator” components** over WebSocket connections. In essence, the environment (which could be a physics simulator, game, or even a physical robot) can run on a local device or separate host, while the RL training algorithm runs on RemoteRL’s cloud infrastructure.

**Key Characteristics:**

* *Separation of Concerns:* The **RL environment (“Simulator”) runs externally** (e.g. on your PC, robot, or another server) and connects to the RemoteRL cloud. The **learning algorithm (“Trainer”) runs in the cloud**, receiving observations and rewards from the simulator and sending back actions or updated policies. This separation means you **don’t need to install or run the environment on the training server**, simplifying setup and leveraging local hardware for simulation if needed.
* *Use Cases:* RemoteRL is especially useful for scenarios like **robotics and IoT** (where a physical robot or device provides real-time data to train a model remotely) and **distributed simulation** (where many simulators run in parallel on different machines to speed up training). It allows training "from anywhere" – whether the environment is on-premise or in the cloud – by handling the networking, synchronization, and scaling behind the scenes. For example, a robotics engineer could connect a robot’s control loop to RemoteRL and have a cloud-based RL agent train on the live data, or a researcher could run dozens of game simulations on local PCs all feeding into a single cloud RL learner.

## Core Features and Functionality

RemoteRL provides a number of features to make remote and distributed RL **easy to integrate and manage**:

* **Zero-Setup Integration:** It works out-of-the-box with popular Python RL libraries. Users simply install the `remoterl` Python package and add a one-line initialization with their API key to an existing Gymnasium or RLlib script – *“Run `pip install remoterl` and add `remoterl.init(API_KEY="…")` to any Gymnasium / Ray RLlib script, and hit run — no rewrites, zero friction”*. This means **no extensive code refactoring or complex cluster setup** is required; RemoteRL hooks into the RL frameworks to handle remote environment communication transparently.

* **Framework and Tool Support:** The service **auto-integrates with popular RL frameworks** like **OpenAI Gymnasium (Gym)** for environment interfaces and **Ray RLlib** for distributed training algorithms. It also supports integration with libraries like **Stable-Baselines3**. This broad support allows developers to use familiar APIs and algorithms (e.g. RLlib’s trainers or Stable-Baselines3’s agents) while RemoteRL manages the remote execution of environments. Whether you use custom Gym environments or standard ones, or whether you train with RLlib’s scalable architecture or a single-agent approach, RemoteRL can plug in with minimal changes.

* **Live Web Dashboard & Monitoring:** RemoteRL provides a **web‑based dashboard** that shows which simulators and trainers are online alongside basic byte/step counters. When you add `remoterl.init(API_KEY="…")` to a process, it appears on the dashboard within seconds. The dashboard displays **connection status only**—RemoteRL does **not** view, store, or inspect observation, reward, or model payloads. Operators can disconnect a simulator or temporarily block new sessions, and any configuration changes apply the next time a pipeline starts; they cannot peek into in‑flight data nor pause individual training iterations.
  
* **Distributed & Scalable Training:** RemoteRL is designed for **scaling to multiple simulators and global use**. It enables easily running **distributed RL** where many environment instances (simulators) feed experience to a central trainer. Under the hood it leverages a scalable cloud backend (built around Ray RLlib for distributed training) to handle synchronization and learning from many parallel environments. A standout feature is *“latency-aware autoscaling — simulators launch in the nearest region, so every env.step() your trainer calls returns faster with minimal round-trip”*. In practice, this means RemoteRL will connect your environment to the closest server region available, reducing network latency and enabling near real-time interactions even if trainer and environment are far apart. The service is **rolling out servers in multiple major cities/regions** to support this. It can therefore support worldwide training setups (e.g. a user in Europe training with simulators in Asia and America concurrently, with each simulator connecting to a nearby node).

* **Real-Time Online RL:** RemoteRL is optimized for **online reinforcement learning**, where training occurs concurrently with data collection (as opposed to offline RL on a static dataset). By streaming observations and rewards from the simulator to the cloud and actions back in real-time, it enables learning “on the fly.” The framework emphasizes minimal latency and continuous training updates – effectively creating a **cloud-like infrastructure specifically tailored for RL workflows** across your devices. This is useful for scenarios requiring immediate feedback, such as training agents in interactive simulations or controlling robots in real time.

* **Usage-Based Pricing and Tiers:** RemoteRL offers a **free tier** and a premium plan. The service will be *“available free for everyone”* for light use, meaning you can run training jobs without paying as long as data usage is under a certain limit. The **free tier includes 1 GB of data traffic free** (as noted in the feature list) and likely some reasonable number of simulators. For heavier usage, **premium services (pay-as-you-go)** are offered – *“charged per GB”* for data beyond the free quota. The premium tier is designed for those who need **“extensive data usage, multiple simulators, and large-scale training across global regions.”** In other words, large experiments or enterprise deployments with many environments and high throughput would incur a usage-based fee. This pricing model is **traffic/duration-based** (billing on both bytes of data exchanged and active connection minutes), ensuring you only pay for what you use, and there is **no hard limit on speed or number of simulators in premium** aside from practical scaling limits. This makes the service accessible to individual researchers (on the free tier) while scaling up to industrial projects on the paid tier.

## Architecture and How It Works

At a high level, RemoteRL follows a **client–server architecture specialized for RL**. The **core components** are defined as follows:

* **Trainer:** The RL algorithm process that you run and manage (e.g., on your own cloud instances, on-prem servers, or a laptop). RemoteRL simply provides the networking bridge: it relays state/reward data from simulators to your trainer and returns actions. You can still scale the trainer across multiple machines—using Ray RLlib or similar frameworks—while RemoteRL abstracts away the connection details, not the computation itself.

* **Simulator:** The environment process you run and whether it’s a game engine, robotics simulator, custom code, or a real-world device such as a robot control loop or IoT sensor. It can live anywhere: on a local machine, an edge device, or another cloud. The simulator connects to RemoteRL over a secure WebSocket, streaming observations and rewards to your trainer and receiving actions in return. Functionally it behaves like any Gym environment; the only difference is that these interactions travel across the network through RemoteRL’s bridge.

```python
import gymnasium as gym
import remoterl 

remoterl.init(api_key="YOUR_KEY", role="trainer")
env = gym.make("Humanoid-v5")  # this environment runs remotely in another city
```

**Communication:** When you integrate RemoteRL, the typical RL loop is split between the trainer and simulator via the network:

* The **trainer invokes** an environment step (e.g. calling `env.step(action)` in RL code). Instead of running locally, this call is routed through RemoteRL’s library which sends the action to the remote simulator over the WebSocket connection.
* The **simulator receives** the action, applies it (advancing the simulation or executing on the robot), and returns the resulting observation, reward, and done flag back over the socket.
* The trainer then uses the received data to update the agent’s model (e.g., via policy gradient, Q-learning update, etc.), and the cycle repeats.
