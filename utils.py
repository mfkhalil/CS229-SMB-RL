import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation

MODEL_DIR = './models/PPO'
VIDEO_DIR = '~/videos/PPO'
LOG_DIR = '~/logs/PPO'


def make_env(grayscale=True, framestack=True):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    if framestack:
        env = VecFrameStack(env, 4, channels_order='last')
    return env


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
            self.model.save(self.models_path)

            # Create a new env for recording
            env = make_env()

            video_path = os.path.join(self.videos_path, 'video_' + str(self.n_calls))
            env = Monitor(env, self.videos_path, force=True, video_callable=lambda episode: True)

            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, _, done, _ = env.step(action)

            env.close()

        return True
