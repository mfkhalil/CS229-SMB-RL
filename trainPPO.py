import argparse
import utils
from stable_baselines3 import PPO

# Define and parse command line arguments
parser = argparse.ArgumentParser(description="Training script for PPO.")
parser.add_argument('--log_dir', type=str, default='./logs/PPO/PPO2', help="Directory for the logs.")
parser.add_argument('--learning_rate', type=float, default=0.0003, help="Learning rate for the PPO algorithm.")
args = parser.parse_args()

env = utils.make_env(log_dir=args.log_dir)
callback = utils.CallbackFunction(models_path=utils.MODEL_DIR, save_frequency=10000)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=utils.LOG_DIR, learning_rate=args.learning_rate, n_steps=512)
model.learn(total_timesteps=1000000, callback=callback)

