# RemoteRL: Seamless Integration of Local Environments with Cloud-Based RL Training

---

## Overview

RemoteRL is a local hosting environment designed for distributed reinforcement learning frameworks such as RLlib. It allows you to run reinforcement learning training jobs in the cloud, while conveniently hosting your environment simulators locally. By seamlessly connecting local environments to cloud-based training on AWS SageMaker, RemoteRL facilitates efficient data collection, rapid experimentation, and scalable multi-agent reinforcement learning workflows.

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

- **Train your RL model on AWS:**
  ```bash
  remoterl train
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
