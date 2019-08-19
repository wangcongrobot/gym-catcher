import gym
import gym_catcher
import numpy as np

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise

env = gym.make("UR5KeepUp-v0")

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
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
# model = HER('MlpPolicy', env, DDPG, n_sampled_goal=n_sampled_goal,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             actor_lr=1e-3, critic_lr=1e-3, action_noise=action_noise,
#             gamma=0.95, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]))


model.learn(int(2e5))
model.save('ur5_keep_up_new_env')

# Load saved model
model = HER.load('ur5_keep_up_new_env', env=env)

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
