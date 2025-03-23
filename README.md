# RemoteRL: WebServer-Based Remote Training for Ray RLlib

---

## Overview

**RemoteRL** simplifies remote reinforcement learning (RL) training specifically tailored for popular RL frameworks such as Ray's RLlib. Leveraging a WebSocket-based interface, RemoteRL seamlessly bridges local RLlib environment simulations with cloud-based training on AWS SageMaker, enabling efficient, scalable, and streamlined RL workflows.

RemoteRL provides two user-friendly interfaces:
- **Command-Line Interface (CLI):** For rapid experimentation and quick validations.
- **Python Library Interface:** For deeper integration and advanced RLlib configurations.

## Installation

Install RemoteRL via pip:

```bash
pip install remoterl --upgrade
```

## Quick Start

### Command-Line Interface

1. **Simulate your local environment on the RemoteRL server:**

   Quickly set up your local environment and automatically connect it to the RemoteRL cloud server for real-time interaction:

   ```bash
   remoterl simulate
   ```

2. **Train your RLlib algorithm on AWS SageMaker:**

   With your environment ready, seamlessly initiate cloud-based RL training:

   ```bash
   remoterl train
   ```

### Python Library Interface

For advanced configurations and programmatic control, use RemoteRL's Python API integrated with Ray RLlib:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from remoterl import RemoteConfig

config = PPOConfig().training(train_batch_size=64)
remote_config = RemoteConfig(config=config)

remote_config.simulate(env="CartPole-v1", region="us-east-1")
remote_config.sagemaker(role_arn="your-role-arn", output_path="s3://your-output-path")
remote_config.train()
```

## Key Benefits

- **Streamlined Workflow:** Effortlessly switch from local testing to cloud-based RL training.
- **Seamless Integration:** Directly compatible with Ray RLlib environments and configuration settings.
- **Optimized Connectivity:** Real-time data communication via WebSockets for efficient environment-agent interactions.

---