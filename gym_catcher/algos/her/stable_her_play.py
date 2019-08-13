"""
train the ur5 keep up env using stable baselines her algorithm
"""
import gym
import gym_catcher 
import numpy as np
import matplotlib.pyplot as plt 

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.results_plotter import load_results, ts2xy


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the tast to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timessteps')
    plt.ylabel('Reward')
    plt.title(title + " Smoothed")
    plt.show()


env = gym.make('UR5KeepUp-v0')

# Load saved model
model = HER.load('sac', env=env)

obs = env.reset()
# env.render('human')
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
# plot_results('/home/cong/workspace/DHER/gym-catcher/gym_catcher/algos/her/data/700')