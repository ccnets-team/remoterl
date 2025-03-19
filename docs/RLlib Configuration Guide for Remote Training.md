# RLlib Configuration Guide for RemoteRL Cloud Training

RemoteRL follows the familiar RLlib configuration template while extending it with an automated conversion and cloud reassembly process. In other words, you build your RLlib configuration as usual, and we convert it into a dictionary (using the `to_dict()` method, as commonly used in RLlib) that’s then reassembled in the cloud. This process preserves each section of your configuration—be it environment settings, training parameters, or network hyperparameters—so that training can seamlessly continue in the cloud.

> **Tip:** For a complete list of configurable parameters and in-depth details, refer to the official [RLlib Algorithm Configuration Documentation](https://docs.ray.io/en/latest/rllib/package_ref/algorithm-config.html).

---

## How It Works

1. **Local RLlib Configuration:**  
   You define your RLlib configuration using standard parameters and methods via the Ray RLlib API. For example, you might set the batch size, learning rate, and rollout fragment lengths just as you would in a regular RLlib setup.

2. **Passing RLlib Config to Cloud Trainer:**  
   RemoteRL uses the `to_dict()` method to export your configuration into a dictionary. This conversion process collects all the modifications you’ve made and carries them to the cloud—whether via the SageMaker hyperparameters section or through a WebSocket channel—to ensure the trainer receives your configuration in real time.

3. **Cloud Reassembly:**  
   In the cloud, the exported dictionary is used to reassemble the complete configuration. This guarantees that your training job in AWS SageMaker (or any supported cloud service) uses the exact settings you configured locally.

---

## Examples

### Example 1: Creating and Exporting an RLlib Configuration

Below is a sample Python snippet showing how you might create an RLlib configuration and export it using RemoteRL’s approach:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls

# Adjust some training parameters using PPOConfig
algorithm_config = (
    PPOConfig()
    .get_default_config()
    .training(train_batch_size=2048, num_sgd_iter=10, lr=5e-4)
    .env_runners(rollout_fragment_length='auto', sample_timeout_s=60)
    .evaluation(num_episodes=10, evaluation_interval=5)
)

from remoterl.config.rllib import RemoteRLlibConfig

# Create a new RLlib configuration instance using RemoteRLlibConfig
rllib_config = RemoteRLlibConfig.from_config(algorithm_config)

# Set up simulation parameters (e.g., for local debugging or environment verification)
rllib_config.simulate(
    env_type='gym',
    env="CartPole-v1",
    num_env_runners=1,
    num_envs_per_env_runner=1,
    region="us-west-2"
)

# Retrieve the remote training key (if assigned)
print("Remote Training Key:", rllib_config.remote_training_key)

# Export the configuration to a dictionary for cloud training
sagemaker_hyperparameters = rllib_config.to_dict()

print("SageMaker Hyperparameters for Cloud Training:")
print(sagemaker_hyperparameters)
```

### Example 2: Using Custom Settings with RemoteRL

In this example, you can see how developers can set their own hyperparameters. The exported configuration will include only the modified parameters, ensuring that your cloud training job is efficient and exactly tailored to your needs.

```python
# Assume you have already created a RemoteRLlibConfig instance (rllib_config) as in Example 1

# For instance, update network settings:
rllib_config.algorithm_config.model(num_layers=5, fcnet_hiddens=[256, 256], use_lstm=False)

# Export the updated configuration
custom_hyperparameters = rllib_config.to_dict()

# These hyperparameters are then used to launch your cloud training job seamlessly.
print("Custom Hyperparameters Ready for Cloud Deployment:")
print(custom_hyperparameters)
```

---
