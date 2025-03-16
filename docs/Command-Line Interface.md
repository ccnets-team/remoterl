# Command-Line Interface

The AgentGPT CLI provides a set of commands to interact with the application directly from your terminal. It makes it easy to configure, simulate, train, and deploy your multi‑agent reinforcement learning environments.

## Commands

### Help
Display help information and usage guidelines for AgentGPT CLI commands.
```bash
agent-gpt --help
agent-gpt config --help
agent-gpt simulate --help
...
```

### Config
Update configuration settings that are used by subsequent training or inference commands.

**Basic Configuration:**  
Update global settings such as hyperparameters and AWS Sagemaker setting.
```bash
# Update global configuration with hyperparameters and SageMaker settings
agent-gpt config --batch_size 256
agent-gpt config --region us-east-1 --role_arn arn:aws:iam::123456789012:role/AgentGPTSageMakerRole
```

**Advanced Module Configuration:**  
For users who wish to fine-tune or override settings—such as environment hosts, exploration methods, or simulator registry details—the CLI also supports advanced commands.  
> **Note:**  
> In most cases, the simulation command automatically configures environment hosts for you. Advanced users can manually adjust these settings if needed.

```bash
# Configure exploration methods for continuous or discrete control.
agent-gpt config exploration set continuous --type gaussian_noise
agent-gpt config exploration set discrete --type epsilon_greedy
agent-gpt config exploration del discrete
```

**Nested Configuration:**  
Update deeply nested configuration parameters using dot notation for fine-grained control.
```bash
# Change the initial sigma for continuous exploration
agent-gpt config --exploration.continuous.initial_sigma 0.2 

```

### List
View the current configuration.
```bash
agent-gpt list
```

### Clear
Reset the configuration cache and CLI state.
```bash
agent-gpt clear
```

### Simulate
Launch simulation environments.  
When you run the `simulate` command, your local machine automatically connects to our AgentGPT WebSocket server on the cloud. This real-time connection enables seamless data communication between your environment's state and the cloud training actions, ensuring that everything is ready for the next agent-gpt train command.

```bash
# Launch a local gym simulation
agent-gpt simulate
```

### Train
Initiate a training job on AWS SageMaker.  
This command uses your configuration (including hyperparameters and sagemaker configurations) to submit a training job to the cloud.
```bash
agent-gpt train
```

### Infer
Deploy or reuse a SageMaker inference endpoint using your stored configuration.  
This command deploys your trained model so that agents can run on AWS.
```bash
agent-gpt infer
```

---
