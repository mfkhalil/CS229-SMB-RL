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

MODEL_DIR = './models/A2C/'
LOG_DIR = './logs/A2C/'

def make_env(grayscale=True, framestack=True, log_dir=LOG_DIR):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    if framestack:
        env = VecFrameStack(env, 4, channels_order='last')
    return env


class CallbackFunction(BaseCallback):

    def __init__(self, models_path, save_frequency, verbose=1):
        super(CallbackFunction, self).__init__(verbose)
        self.models_path = models_path
        self.save_frequency = save_frequency

    def _init_callback(self):
        if self.models_path is not None:
            os.makedirs(self.models_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_frequency == 0:
            model_path = os.path.join(self.models_path, 'best_model_' + str(self.n_calls))
            self.model.save(model_path)

        return True
