import gymnasium as gym
from typing import Any, Tuple

class SecureWrapper(gym.Wrapper):
    """Wrapper that automatically encrypts/decrypts observations and rewards.

    Leave :meth:`encrypt` and :meth:`decrypt` as-is if encryption is not required.
    """

    # ── Default implementation: no‑op ─────────────────────────
    def encrypt(self, data: Any) -> Any:      # simply return the original
        return data

    def decrypt(self, data: Any) -> Any:      # simply return the original
        return data
    # ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.encrypt(obs), info

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, dict]:
        dec_action = self.decrypt(action)
        obs, reward, terminated, truncated, info = self.env.step(dec_action)
        return (
            self.encrypt(obs),
            self.encrypt(reward),
            terminated,
            truncated,
            info,
        )
    
    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space     # pass-through  

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space          # pass-through
    