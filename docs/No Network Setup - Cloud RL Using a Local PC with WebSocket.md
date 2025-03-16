# No Network Setup - Cloud RL Using Local PC with WebSocket

This document introduces a novel approach to directly connect your local PC to a cloud server and start online reinforcement learning (RL) without requiring any network configuration on your part. By leveraging AWS WebSocket API, this solution bypasses traditional connection methods that require manual network setup—such as port forwarding for local IP connections or using tunneling services like ngrok.

---

## Overview

Cloud-based RL training has demanded extensive knowledge of network configuration to enable local environments to interact with cloud services. This new method eliminates those barriers by establishing a direct, automated connection between your local simulation environment and the cloud trainer. Key benefits include:

- **No Manual Network Setup & Installation:**  
  For local to cloud communication for RL training, this method simplifies the process.
  There is no need to configure routers, manage IP addresses, or set up tunnels—eliminating the need for ngrok, firewall adjustments, or port forwarding.

- **Automated Configuration:**  
  With this method, network-related hyperparameters (such as environment hosts or endpoints) are not required. The WebSocket API automatically establishes bidirectional communication between your environment and the environment gateway located in our cloud trainer on AWS SageMaker.

- **Easy Entry for Cloud RL Training:**  
  Provides a straightforward entry point for initiating cloud-based RL training directly from your local environment.

- **Scalable and Future-Proof:**  
  Soon to be fully operational with our WebSocket API integrated with Lambda servers for efficient scaling.

---

## Architecture

### WebSocket Communication via WSEnvAPI

At the core of this solution is the **WSEnvAPI** [https://github.com/ccnets-team/agent-gpt/blob/main/agent_gpt/env_host/env_api_websocket.py] a remote gym module that facilitates WebSocket-based communication between your local PC and the cloud trainer. Key components include:


- **Synchronous/Asynchronous Methods for Environment Control:**
  - **make / make_vec:** Create single or multiple simulation environments.
  - **reset:** Initialize or reset an environment.
  - **step:** Process actions in the environment and return resulting states.
  - **close:** Terminate a simulation environment.
  - **get_action_space / get_observation_space:** Retrieve details about the environment's configuration.


---

## How It Works

1. **Establishing the Connection:**
   - Your local simulation environment initiates a WebSocket connection provided by WSEnvAPI.

2. **Processing and Response:**
   - The synchronous methods in WSEnvAPI process commands, manage the simulation environments, and send real-time responses back to the cloud trainer.

3. **Automated Host Configuration:**
   - The system automatically updates environment host details, eliminating the need for manual network configuration (e.g., IP setups or tunnels).

---

## Future Enhancements

- **Expanded Lambda Integration:**  
  Full integration with Lambda servers will enable more efficient data communication that reduce latency.

- **Enhanced Monitoring Tools:**  
  Upcoming updates will include advanced logging and real-time monitoring dashboards.

- **Scalability Improvements:**  
  As demand increases, additional scalability features will support larger training workloads and multiple concurrent simulations.

---

## Summary

This setup-free approach allows cloud-based RL training to eliminate internet & network configuration requirements. With automated WebSocket communication and dynamic environment host configuration, you can focus on training and simulation without the overhead of manual network setup. Stay tuned as we roll out full operational support with our WebSocket API, aiming for easiler cloud RL training ecosystem.

For further details, configuration options, and troubleshooting, please refer to our related guides:
- [Command-Line Interface](./Command-Line%20Interface.md)
- [Local Environment Hosting for Cloud Training - Port Forwarding & Tunneling](./Local%20Environment%20Hosting%20for%20Cloud%20Training%20-%20Port%20Forwarding%20&%20Tunneling.md)]
