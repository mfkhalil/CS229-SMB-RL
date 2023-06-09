import utils
from stable_baselines3 import PPO


env = utils.make_env()
callback = utils.CallbackFunction(models_path=utils.MODEL_DIR, save_frequency=10000)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=utils.LOG_DIR, learning_rate=0.0003, n_steps=512)
model.learn(total_timesteps=1000000, callback=callback)
