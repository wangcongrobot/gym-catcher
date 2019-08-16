"""
train the ur5 keep up env using stable baselines her algorithm
"""

import numpy as np
import gym
import gym_catcher
from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise

from stable_baselines.logger import configure
configure()

env = gym.make('UR5KeepUp-v0')

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            tensorboard_log="./log_stable_her/",
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))

# DDPG Hyperparams:
# NOTE: it works even without action noise
# n_actions = env.action_space.shape[0]
# noise_std = 0.2
#

model.learn(int(2e5))
model.save('her_sac_ur5_keep_up')

# Load saved model
model = HER.load('her_sac_ur5_keep_up', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(100):
  action, _ = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
      print("Reward:", episode_reward, "Success?", info.get('is_success', False))
      episode_reward = 0.0
      obs = env.reset()
