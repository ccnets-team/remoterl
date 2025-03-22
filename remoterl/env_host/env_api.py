from websocket._exceptions import WebSocketTimeoutException, WebSocketConnectionClosedException
import numpy as np
import logging
import websocket
import json
import socket
import threading
from typing import Optional, Any
import msgpack
import base64

# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from ..utils.conversion_utils import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
    replace_nans_infs,
    space_to_dict,
)
from ..utils.message_utils import (
    default, 
    slice_data,
)

WEBSOCKET_TIMEOUT = 1
class EnvAPI:
    def __init__(self, env_wrapper, remote_training_key, remote_rl_server_url, 
               env_idx, num_agents):
        self.env_wrapper = env_wrapper
        self.environments = {}
        self.env_idx = env_idx
        self.shutdown_event = threading.Event()
        self.ws = websocket.WebSocket()
        self.cnt_msg = 0
        self.msg_print_interval = 100
        self.max_print_length = 200
        print("Connecting to Remote RL server..., ", remote_rl_server_url)
        self.patience = 120
        self.patience_threshold = 120
        self.num_slices = 1
        self.ws.connect(remote_rl_server_url)
        self.ws.settimeout(WEBSOCKET_TIMEOUT)
        
        self.send_message("init", remote_training_key, data = {"env_idx": env_idx, "num_agents": num_agents})
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.ws:
            print("Closing WebSocket connection.")
            self.ws.close()
        for env_key in list(self.environments.keys()):
            self.environments[env_key].close()  
            del self.environments[env_key]
    
    def check_alive(self):
        self.patience += 1
        if self.patience > self.patience_threshold:
            heartbeat_message = (
                f"No training activity detected. Environment {self.env_idx} is still online and waiting..."
            )            
            if self.env_idx == 0:
                print("Sending heartbeat: ", heartbeat_message)
            self.send_message("event", message=heartbeat_message, type="heartbeat")
            self.patience = 0         
              
    def communicate(self):
        message_buffer = {}

        while not self.shutdown_event.is_set():
            try:
                packed_request = self.ws.recv()
                self.patience = 0
            except (socket.timeout, WebSocketTimeoutException):
                self.check_alive()
                continue
            except WebSocketConnectionClosedException:
                logging.warning("WebSocket connection closed by server.")
                break
            except Exception as e:
                logging.exception("WebSocket receiving error: %s", e)
                continue

            try:
                if packed_request.count(':') == 3:
                    slice_idx, total_slices, method, slice_data = packed_request.split(':', 3)
                    slice_idx = int(slice_idx)
                    total_slices = int(total_slices)

                    if method not in message_buffer:
                        message_buffer[method] = [None] * total_slices

                    message_buffer[method][slice_idx] = slice_data

                    if None in message_buffer[method]:
                        continue  # Wait for all slices

                    complete_packed = ''.join(message_buffer[method])
                    del message_buffer[method]  # Clear buffer after completion
                else:
                    complete_packed = packed_request
                payload = self.unpack_request(complete_packed)
                
                data = payload.get("data", {})
                
                data_str = repr(data)
                if len(data_str) > self.max_print_length:
                    data_str = data_str[:self.max_print_length] + " ... [truncated]"

                if self.cnt_msg % self.msg_print_interval == 0:
                    print(
                        f"[Msg {self.cnt_msg:05d}] Received request:\n"
                        f"    Method : {method}\n"
                        f"    Num Msg Slices: {self.num_slices}\n"
                        f"    Data   : {data_str}"
                    )
                self.cnt_msg += 1                
                
                method = data.get("method")
                env_key = data.get("env_key")
                # Execute method based on request
                if method == "make":
                    result = self.make(env_key, data.get("env_id"), data.get("render_mode"))
                elif method == "make_vec":
                    result = self.make_vec(env_key, data.get("env_id"), data.get("num_envs"))
                elif method == "reset":
                    result = self.reset(env_key, data.get("seed"), data.get("options"))
                elif method == "step":
                    result = self.step(env_key, data.get("action"))
                elif method == "close":
                    result = self.close(env_key)
                elif method == "observation_space":
                    result = self.observation_space(env_key)
                elif method == "action_space":
                    result = self.action_space(env_key)
                else:
                    result = self.send_message("event", message=f"Unknown method: {method}")
                    
                self.send_response(result, method)

            except Exception as e:
                logging.exception("Error processing message: %s", e)
                self.send_message("event", message=f"Internal server error: {str(e)}", type="error")
                continue

    def send_response(self, result, method):
        packed = self.pack_response(result)
        slices = slice_data(packed, method)
        self.num_slices = len(slices)
        for i, data_slice in enumerate(slices):
            self.ws.send(data_slice)
            
    def pack_response(self, result):
        packed = msgpack.packb(result, use_bin_type=True, default=default)
        packed_response = base64.b64encode(packed).decode('utf-8')
        return packed_response

    def unpack_request(self, packed_request):
        packed_payload = base64.b64decode(packed_request)
        payload = msgpack.unpackb(packed_payload, raw=False)
        return payload
    
    def send_message(self, action, remote_training_key=None, data=None, message=None, type="info"):
        payload = {"action": action}
        if remote_training_key is not None:
            payload["training_key"] = remote_training_key
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        if type is not None:
            payload["type"] = type
            
        self.ws.send(json.dumps(payload))

    # ----------------- Environment methods -----------------

    def make(self, env_key: str, env_id: str, render_mode: Optional[str] = None):
        env_instance = self.env_wrapper.make(env_id, render_mode=render_mode)
        self.environments[env_key] = env_instance
        return {"message": f"Environment {env_id} created.", "env_key": env_key}

    def make_vec(self, env_key: str, env_id: str, num_envs: int):
        env_instance = self.env_wrapper.make_vec(env_id, num_envs=num_envs)
        self.environments[env_key] = env_instance
        return {"message": f"Vectorized environment {env_id} created.", "env_key": env_key}

    def reset(self, env_key: str, seed: Optional[int], options: Optional[Any]):
        env = self.environments[env_key]
        observation, info = env.reset(seed=seed, options=options)
        return {"observation": convert_ndarrays_to_nested_lists(observation), "info": convert_ndarrays_to_nested_lists(info)}

    def step(self, env_key: str, action_data):
        env = self.environments[env_key]
        action = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        return {
            "observation": convert_ndarrays_to_nested_lists(observation),
            "reward": convert_ndarrays_to_nested_lists(reward),
            "terminated": convert_ndarrays_to_nested_lists(terminated),
            "truncated": convert_ndarrays_to_nested_lists(truncated),
            "info": convert_ndarrays_to_nested_lists(info)
        }

    def action_space(self, env_key: str):
        return replace_nans_infs(space_to_dict(self.environments[env_key].action_space))

    def observation_space(self, env_key: str):
        return replace_nans_infs(space_to_dict(self.environments[env_key].observation_space))

    def close(self, env_key: str):
        if env_key in self.environments:
            self.environments[env_key].close()
            del self.environments[env_key]
            return {"message": f"Environment {env_key} closed."}
        return {"error": f"Environment {env_key} not found."}