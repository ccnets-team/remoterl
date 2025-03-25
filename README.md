# RemoteRL: Remote Environment-Integrated Online RL

## Overview

**RemoteRL** enables remote reinforcement learning (RL) training specifically tailored for popular frameworks like Gymnasium or Ray's RLlib. By utilizing a WebSocket-based real-time training interface, RemoteRL seamlessly with training infrastructure running either locally or on the cloud. This approach leverages the advantages of moving from offline to online training, offering immediate updates, improved scalability, efficient resource management, and streamlined RL workflows.

---

## 🟢 Quick Start Guide

Select the setup method that best matches your requirements:

### ① Environment Setup

Configure and integrate your environment simulations to connect seamlessly with our web server, enabling real-time communication between the environment's states and RL algorithms' actions.

- **Command-line Interface (CLI)**  
  Ideal for quick simulations, educational demonstrations, and debugging.

  **Quick Usage**:
  ```bash
  pip install remoterl --upgrade
  remoterl simulate
  remoterl train
  ```

- **Python Library**  
  Integrate RemoteRL directly into your Python workflows for automation, scripting, and moderate customization.

  **Quick Usage**:
  ```bash
  pip install remoterl --upgrade
  ```

  ```python
  from ray.rllib.algorithms.ppo import PPOConfig
  from remoterl import RemoteConfig

  config = PPOConfig().training(train_batch_size=64)
  remote_config = RemoteConfig(config=config)

  remote_config.simulate(env="CartPole-v1", region="us-east-1")
  remote_config.sagemaker(
      role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
      output_path="s3://your-output-path"
  )
  remote_config.train()
  ```

- **Docker Container** *(Coming Soon)*  
  Ready to use entering entry_point of your environment(eq, <module>:<environment_class>)

  **Example Usage (Planned Feature)**:
  ```bash
  docker run -v ${env_dir}:/envs remoterl-env:ray /envs/${entry_point}
  ```

---

### ② Trainer Pairing

Effortlessly pair your reinforcement learning algorithms with your environments via our web server.

- **RemoteRL Container(PyTorch-Ray on SageMaker)**  
  Through our CLI or Python library import, you can easily start training:

  ```bash
  pip install remoterl --upgrade
  remoterl train
  ```

  ```python
  from ray.rllib.algorithms.ppo import PPOConfig
  from remoterl import RemoteConfig

  remote_config = RemoteConfig(config=PPOConfig())
  # Additional configuration steps...
  remote_config.train()
  ```
  
- **Client’s Own Container**
  Fully customizable containers that integrate with RemoteRL by including our Python package during your Docker container build process.

  **Quick Docker Example** *(customizable; base image below is illustrative only)*:

  ```Dockerfile
  # Use any suitable base image (this is an illustrative example)
  FROM python:3.12-slim

  # Install RemoteRL trainer and additional dependencies
  RUN pip install remoterl-trainer[ray] your-other-lib1 your-other-lib2

  # Add your custom commands here
  ```

  Build your container:
  ```bash
  docker build -t your-custom-trainer:latest .
  ```

---

### ③ Environment Hosting (Environment-to-Trainer Ratio)

Defines how many environment simulation instances (local or remote) connect concurrently to a single trainer instance:

- **Single Environment (1-to-1)**  
  Ideal for prototyping, demos, and small-scale tests.

- **Multiple Environments (N-to-1)** *(Coming Soon)*  
  Designed for scalability, parallel training, faster iteration, and industrial-level applications.

---

### ④ Deployment Scenario

Determines where your environments and trainers run:

- **Local-to-Local**  
  Both environments and trainer run on local hardware, suited for quick tests and debugging.

- **Local-to-Cloud**  
  Environments run locally while training occurs in the cloud, optimizing resource usage and cost efficiency.

- **Cloud-to-Cloud**  
  Environments and trainer fully utilize cloud infrastructure, ideal for enterprise-level scalability and extensive research.

---

## ✅ RemoteRL Categorization Matrix

Choose your optimal setup using this clear matrix:

| Environment Interface | Trainer Interface                               | Environment Hosting | Deployment Scenario                              | Recommended Use Case                                         |
|-----------------------|-------------------------------------------------|---------------------|---------------------------------------------------------------|--------------------------------------------------|
| CLI                   | RemoteRL Container                                | 1-to-1              | Local-to-Cloud                 | Beta-testing              |
| Python Library        | RemoteRL Container                                | 1-to-1              | Local-to-Cloud                 | Quick prototyping, Debugging, Experiments            |
| Docker Container      | Client Container      | N-to-1              | Local/Cloud-to-Local/Cloud         | Scalable production & deployments    |

---

