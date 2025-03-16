
# Hyperparams Guide

---

### Session Settings

- **use_graphics**  
  *Flag to render environment graphics during training.*  
  - In our system, the environment simulation runs on the client side (either on a local machine or a cloud provision). Client environments communicate with our remote cloud trainer, so simulators are not hosted in the cloud.  
  - This flag is primarily intended for local debugging and visual validation, allowing the cloud trainer to instruct the local machine to display simulation graphics for debugging purposes.

- **resume_training**  
  *Indicator to resume training from a checkpoint.*  
  - If enabled, the trainer loads saved state and weights, ensuring continuity in long or interrupted training sessions.

---

### Training Parameters

- **batch_size**  
  *Number of samples per gradient update.*  
  - Larger batch sizes reduce gradient variance but increase memory usage.
  - Balancing batch size with buffer capacity and max input states is crucial. While a larger batch size yields more stable gradients, the max input states parameter ensures sufficient context is provided, thereby reducing bias in the training data.

- **buffer_size**  
  *Capacity of the experience replay buffer.*  
  - Determines the number of transitions stored; a larger buffer offers a more diverse training set at the expense of higher memory requirements.

- **replay_ratio**  
  *Ratio of training updates to environment interactions.*  
  - Controls how frequently the policy is updated relative to the rate of data collection.
  - In our system, workers gather transitions from local environments over the internet. The replay ratio helps manage the data collection speed, ensuring training remains robust regardless of internet conditions.

- **max_steps**  
  *Maximum number of environment interaction steps.*  
  - Sets the overall training horizon and acts as an experiment termination criterion.
  - Keep in mind potential timeouts (e.g., AWS instance limits) when configuring this parameter.

---

### Algorithm Hyperparameters

- **adaptive_rl_hyperparams**  
  Critical RL parameters (e.g., gamma, lambda, reward scale) are trainable. This design enables the agent to dynamically adjust these values during training to optimize advantage estimation and normalization. These parameters are initialized with default values and updated via gradient-based optimization alongside network weights.

- **gamma_init**  
  *Initial discount factor for RL.*  
  - Determines the weighting of future versus immediate rewards.  
  - The “init” suffix indicates that this value is adaptable during training to better handle long-horizon dependencies.

- **lambda_init**  
  *Initial lambda for generalized advantage estimation in off-policy training.*  
  - Balances bias and variance in advantage estimates.  
  - Like `gamma_init`, this parameter is subject to dynamic adjustment throughout training.

- **max_input_states**  
  *Context window size for the transformer.*  
  - Specifies the number of past states included in each input sequence, analogous to the number of tokens in NLP transformer inputs. Balancing this value with `batch_size` is crucial for maintaining a diverse training set and reducing bias in the training data.  
  - **Note:** Default is set to 16 for G5 instances to accommodate VRAM constraints (24GB per GPU).

- **exploration**  
  *Mapping of exploration strategy configurations.*  
  - Associates strategy identifiers (e.g., "continuous", "discrete") with their specific parameters (e.g., Gaussian noise settings, epsilon decay).  
  - Enables fine-grained control over the exploration-exploitation trade-off during training.

---

### Optimization Hyperparameters
- **lr_init**  
  *Initial learning rate for the optimizer.*  
  - Sets the starting point for learning rate schedules, directly impacting convergence speed.

- **lr_end**  
  *Final learning rate target.*  
  - Works in tandem with the scheduler (e.g., linear, exponential) to dictate the decay of the learning rate throughout training.

- **lr_scheduler**  
  *Learning rate scheduling strategy.*  
  - Defines how the learning rate decays over time, impacting the stability and efficiency of training updates.

- **tau**  
  *Soft update coefficient for target networks.*  
  - Determines the interpolation rate between online and target network parameters.  
  - Lower values lead to smoother, more stable updates, reducing variance in target predictions.

- **max_grad_norm**  
  *Threshold for gradient clipping.*  
  - Prevents gradient explosion by normalizing gradients when their L2 norm exceeds this value.

---

### Network Hyperparameters
- **Unified Causal RL Network Architecture**  
  Critic GPT, Actor GPT, and Reverse-environment GPT share the same settings for `num_layers`, `d_model`, `num_heads`, and `dropout`. This consistency ensures uniform representational capacity, even though each network is trained on different objectives and data sources. Future updates may allow for independent customization.

- **gpt_type**  
  *Specifies the GPT variant (e.g., "gpt2").*  
  - Directly corresponds to models from the Hugging Face Transformers library. In our architecture, the GPT model is customized for three distinct roles: the actor predicts continuous or discrete actions; the critic performs regression to estimate value functions; and the reverse-environment models the reverse transition of agent states. Currently, only GPT-2 is supported, with plans to integrate GPT-3 architectures for enhanced agent action learning in the future. Even within the GPT-2 family, model capacity can be adjusted by tuning `d_model` (hidden state dimensionality) and `num_layers` (number of transformer blocks).

- **num_layers**  
  *Specifies the number of transformer blocks in each GPT model.*  
  - This parameter sets the depth of each individual network. For instance, if `num_layers` is set to 5, then each of the Critic GPT, Actor GPT, and Reverse-environment GPT models will have 5 transformer blocks. This results in an effective total of 15 layers during a forward pass across the three networks. Due to their cooperative operation, even a modest layer count per network can offer high representational capacity. Thus, even a modest layer count per network can offer high representational capacity, potentially making smaller values sufficient for complex tasks compared to traditional deep learning architectures.
  
- **d_model**  
  *Dimensionality of the model’s hidden states and embeddings.*  
  - A higher `d_model` increases the representational capacity, allowing the network to model finer details, albeit with higher computational costs.

- **dropout**  
  *Dropout rate used for regularization.*  
  - Applied across layers to mitigate overfitting, particularly critical in high-capacity models.

- **num_heads**  
  *Number of attention heads in the transformer architecture.*  
  - More attention heads enable the model to capture a diverse set of relationships in the input data, though this comes with increased computational overhead.

---
