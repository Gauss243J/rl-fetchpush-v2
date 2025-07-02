import gym
from stable_baselines3 import DDPG

env = gym.make("FetchPush-v2")

model = DDPG("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
