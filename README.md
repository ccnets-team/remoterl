# RemoteRL: Remote Env Integrated Cloud RL Training

![How RemoteRL Works](https://imgur.com/r4hGxqO.png)
---

## Overview

RemoteRL is a one-click, cloud-based platform for distributed reinforcement learning. It lets you easily host your environment simulators—either locally or in the cloud—and connect them to a central training job on AWS SageMaker. This enables efficient data collection and scalable multi-agent training using rllib.

## Installation

```markdown
pip install remoterl --upgrade
```

### Simulation

- **Launch your environment simulator (e.g., Gym, Unity, Unreal) before training begins:**  
  With this command, your local machine automatically connects to our RemoteRL WebSocket server on the cloud. This real-time connection enables seamless data communication between your environment's state and the cloud training actions, ensuring that everything is ready for the next `remoterl train` command.

  ```bash
   remoterl simulate
  ```

### Training & Inference

- **Train a gpt model on AWS:**
  ```bash
  remoterl train
  ```

- **Run remote rl on AWS:**
  ```bash
  remoterl infer
  ```

### Configuration

- **Config RLLibConfig & SageMaker:**
  ```bash
  remoterl config --batch_size 256
  remoterl config --role_arn arn:aws:iam::123456789012:role/SageMakerExecutionRole
  ```
- **List & Clear current configuration:**
  ```bash
  remoterl list
  remoterl clear
  ```

## Key Features

- **Cloud & Local Hosting:** Quickly deploy environments (Gym/Unity) with a single command.
- **Parallel Training:** Connect multiple simulators to one AWS SageMaker trainer.
- **Real-Time Inference:** Serve a GPT-based RL policy for instant decision-making.
- **Cost-Optimized:** Minimize expenses by centralizing training while keeping simulations local if needed.
- **Scalable GPT Support:** Train Actor (policy) and Critic (value) GPT models together using reverse transitions.
