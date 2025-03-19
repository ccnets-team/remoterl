# RemoteRL: Bridging Local Environments to RLLib Training on Cloud

RemoteRL is our midway project on the journey to [AgentGPT](https://github.com/ccnets-team/agent-gpt). By decoupling the communication module from AgentGPT, RemoteRL is dedicated to enabling fully customizable Ray RLlib training with scalable cloud resources—all while retaining the flexibility and convenience of local environment simulation.

## What is RemoteRL?

RemoteRL seamlessly connects your local simulation environments to robust, cloud-based RL training. Our innovative WebSocket server facilitates efficient, bidirectional data communication between your local machine and the cloud. This smart communication strategy not only slashes cloud provisioning costs but also offers the ease of debugging and customization locally. With real-time training updates and a refined RLlib configuration, you gain granular control over algorithm hyperparameters. In essence, RemoteRL lets you bring your custom settings, while our platform handles the complexities of launching and managing cloud training.

## Key Features

- **Seamless Cloud Integration:**  
  Run your RL agents on powerful cloud instances while maintaining full control over local simulation customization. By managing state and action data exchange between your local environment and the cloud in real time, RemoteRL minimizes unnecessary cloud usage and reduces overall training costs. Its WebSocket API automatically establishes a secure, bidirectional connection between your local PC and the cloud trainer—eliminating the need for complex network setups like routers, port forwarding, or tunneling services.

- **Optimized RLlib Configuration:**  
  Our enhanced RLlib config file leverages Ray RLlib’s algorithm setup to apply only the modified parameters—such as batch size, learning rate, rollout fragment lengths, and network architectures—ensuring efficient and scalable cloud training. Seamlessly integrated with AWS SageMaker settings, this unified approach simplifies launching training jobs and lets you focus on fine-tuning your RL models.

- **Real-Time Training Control (Upcoming):**  
  Debug and fine-tune your RL algorithms in your familiar local environment before deploying to the cloud. Soon, you'll be able to monitor training progress, interrupt iterations, update configurations, and resume from checkpoints—all without restarting the entire RL pipeline.

### Command-Line Interface

RemoteRL’s CLI makes it effortless to configure, simulate, and train:
  
- **Simulation and Training:**  
  Launch local simulations using `remoterl simulate` and submit your training jobs to the cloud with `remoterl train`. Your configuration updates are directly applied to the RLlib algorithm running on AWS SageMaker.

## Getting Started

1. **Install the Package:**
   ```bash
   pip install remoterl
   ```

2. **Simulate:**  
    Launch your local simulation environment:
    ```bash
    remoterl simulate
    ```
    The CLI will prompt you for environment details (e.g., `env_type`, `env`, `num_env_runners`, `num_envs_per_env_runner`, `region`). These settings are sent to our WebSocket server, which then provides a unique training key to initiate a cloud training job.

3. **Train:**  
    Once your simulation is running, start your cloud training job:
    ```bash
    remoterl train
    ```
    During training, your AWS credentials (`role_arn`) and the output path (`output_path`) are automatically used along with the training key provided earlier.

## Conclusion

RemoteRL empowers you to bridge the gap between local customization and robust, scalable cloud RL training. With a refined RLlib configuration integrated seamlessly into AWS SageMaker deployments, you can fine-tune your RL algorithms locally and let RemoteRL manage the cloud training complexities.
```