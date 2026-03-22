"""
Atari Preprocessing Wrappers
Standard DeepMind stack: NoOp → MaxSkip → EpisodicLife → FireReset
                         → WarpFrame → ClipReward → FrameStack
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


# ─────────────────────────────────────────────
# 1. NoopResetEnv
# ─────────────────────────────────────────────
class NoopResetEnv(gym.Wrapper):
    """
    Sample a random number of no-op actions at the start of each episode.
    Prevents the agent from memorising a fixed starting state.
    """
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


# ─────────────────────────────────────────────
# 2. MaxAndSkipEnv
# ─────────────────────────────────────────────
class MaxAndSkipEnv(gym.Wrapper):
    """
    Repeat the same action for `skip` frames.
    Take the pixel-wise max over the last 2 frames to handle Atari flickering.
    Return the sum of rewards over the skipped frames.
    """
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Max-pool over last 2 frames to remove flicker
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ─────────────────────────────────────────────
# 3. EpisodicLifeEnv
# ─────────────────────────────────────────────
class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat each life loss as an episode terminal signal.
    The actual game reset only happens when all lives are exhausted.
    This gives a denser learning signal and makes the agent value survival.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Life was lost — signal terminal without actually resetting
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Step with no-op to advance past life-loss screen
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# ─────────────────────────────────────────────
# 4. FireResetEnv
# ─────────────────────────────────────────────
class FireResetEnv(gym.Wrapper):
    """
    Press FIRE on reset for games that require it to start (e.g. Breakout).
    Safe to apply to Pong too — it simply won't be needed there.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        return self.env.step(action)


# ─────────────────────────────────────────────
# 5. WarpFrame
# ─────────────────────────────────────────────
class WarpFrame(gym.ObservationWrapper):
    """
    Convert RGB frames to 84×84 grayscale using bilinear interpolation.
    This is the standard Atari preprocessing resolution.
    """
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]  # (H, W, 1)


# ─────────────────────────────────────────────
# 6. ClipRewardEnv
# ─────────────────────────────────────────────
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, +1} using np.sign.
    This normalises the reward scale across games, making hyperparameters
    more transferable and training more stable.
    """
    def reward(self, reward):
        return np.sign(reward)


# ─────────────────────────────────────────────
# 7. FrameStack
# ─────────────────────────────────────────────
class FrameStack(gym.Wrapper):
    """
    Stack the last `k` frames along the channel dimension.
    Input shape becomes (k, H, W) — gives the agent motion information.
    Standard k=4: agent sees 4 consecutive grayscale frames.
    """
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = np.zeros(
            (k,) + env.observation_space.shape[:2], dtype=np.uint8
        )
        h, w = env.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = 0
        self.frames[-1] = obs[:, :, 0]
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs[:, :, 0]
        return self.frames.copy(), reward, terminated, truncated, info


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────
def make_atari_env(env_id: str, seed: int = 42, render_mode: str = None) -> gym.Env:
    """
    Build a fully preprocessed Atari environment.

    Wrapper order matters:
        NoopReset → MaxAndSkip → EpisodicLife → FireReset
        → WarpFrame → ClipReward → FrameStack

    Args:
        env_id:      Gymnasium env id, e.g. "ALE/Pong-v5"
        seed:        Random seed for reproducibility
        render_mode: None for training, "human" or "rgb_array" for rendering

    Returns:
        Fully wrapped gymnasium environment
    """
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    return env


if __name__ == "__main__":
    # Quick sanity check
    for game in ["ALE/Pong-v5", "ALE/Breakout-v5"]:
        env = make_atari_env(game, seed=0)
        obs, _ = env.reset()
        print(f"{game}: obs shape = {obs.shape}, action space = {env.action_space.n}")
        for _ in range(5):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
        env.close()
        print(f"  ✓ Wrapper stack OK")
