import os
from collections import deque
import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation, FrameStack
from gym import spaces
from gym.spaces.box import Box

MODEL_DIR = './models/PPO/'
VIDEO_DIR = '~/videos/PPO/'
LOG_DIR = '~/logs/PPO/'


def make_env(grayscale=True, framestack=True):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    if framestack:
        env = VecFrameStack(env, 4, channels_order='last')
    return env

class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(low=np.min(low, axis=0), high=np.max(high, axis=0), 
                                            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.num_stack
        return np.squeeze(np.stack(self.frames, axis=0), axis=-1)

class CallbackFunction(BaseCallback):

    def __init__(self, models_path, videos_path, save_frequency, verbose=1):
        super(CallbackFunction, self).__init__(verbose)
        self.models_path = models_path
        self.videos_path = videos_path
        self.save_frequency = save_frequency

    def _init_callback(self):
        if self.models_path is not None:
            os.makedirs(self.models_path, exist_ok=True)
        if self.videos_path is not None:
            os.makedirs(self.videos_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_frequency == 0:
            model_path = os.path.join(self.models_path, 'best_model_' + str(self.n_calls))
            self.model.save(model_path)

            # Create a new env for recording
            env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = GrayScaleObservation(env, keep_dim=True)
            env = CustomFrameStack(env, num_stack=4)

            video_path = os.path.join(self.videos_path, 'video_' + str(self.n_calls))
            env = Monitor(env, self.videos_path)

            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                action = action.item()
                obs, _, done, _ = env.step(action)

            env.close()

        return True
