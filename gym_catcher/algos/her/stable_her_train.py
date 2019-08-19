"""
train the ur5 keep up env using stable baselines her algorithm
"""

import os
import argparse

import gym
import gym_catcher
import numpy as np
import matplotlib.pyplot as plt

# os.environ['LD_LIBRARY_PATH'] = os.environ['HOME'] + '/.mujoco/mujoco200/bin:'

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.bench import Monitor
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.logger import configure 
from stable_baselines.results_plotter import load_results, ts2xy


tb_logdir = '/home/cong/workspace/DHER/gym-catcher/gym_catcher/algos/her/data/'

# openai baselines monitor using file tb/ to record, tensorboard
configure(folder=tb_logdir)

ENV = "UR5KeepUp-v0"
ALGO = 'ddpg'
NB_TRAIN_EPS = 1000 # corresponding to one HER baseline run for Fetch env
EP_TIMESTEPS = 50
LOG_INTERVAL = 100 # every 2000 episodes

ALGOS = {
    'sac': SAC,
    'ddpg': DDPG,
    # 'dqn': DQN # does not support continuous actions
}

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

def find_save_path(dir, trial_id):
    """
    Create a directory to save results and arguments. Adds 100 to the trial id if a directory already exists.
    Params
    ------
    - dir (str)
        Main saving directory
    - trial_id (int)
        Trial identifier
    """
    i=0
    while True:
        save_dir = dir+str(trial_id+i*100)+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i+=1

    return save_dir

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print states every 1000 calls
    if (n_steps + 1) % 100 == 0:
          # Evaluate policy training performance
          x, y = ts2xy(load_results(logdir), 'timesteps')
          if (len(x) > 0) and (len(y) > 100):
                mean_reward = np.mean(y[-100])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                      best_mean_reward = mean_reward
                      # Example for saving best model
                      print("Saving new best model")
                      _locals['self'].save(logdir + 'best_model.pkl')
    n_steps += 1
    # Return False will stop training early
    return True

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

def launch(algo, env_id, trial_id, seed):

    logdir = find_save_path('./data/' + env_id + "/", trial_id)
    # logdir = '/tmp/her/'

    algo_ = ALGOS[algo]
    env = gym.make(env_id)
    # Wrap the environment in a Monitor wrapper to record training progress
    # Note: logdir must exist
    os.makedirs(logdir, exist_ok=True)
    env = Monitor(env, logdir, allow_early_resets=True)

    eval_env = gym.make(env_id)
    eval_env = Monitor(eval_env, logdir, allow_early_resets=True)
    if not isinstance(env, HERGoalEnvWrapper):
        eval_env = HERGoalEnvWrapper(eval_env)

    if algo_ == SAC:
        kwargs = {'learning_rate': 1e-3}
    elif algo_ == DDPG:
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

        kwargs = {
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'action_noise': action_noise
        }
    else:
        raise ValueError('Algo not supported: {}'.format(algo_))

    model = HER('MlpPolicy', env, algo_, n_sampled_goal=n_sampled_goal, 
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e3), 
                # nb_train_steps=100, 
                # eval_env=eval_env, 
                # nb_eval_steps=20*EP_TIMESTEPS,
                gamma=0.95, batch_size=256,
                # tensorboard_log="./log_stable_her/",
                policy_kwargs=dict(layers=[256, 256, 256]), **kwargs)

    # model.learn(total_timesteps=NB_TRAIN_EPS * EP_TIMESTEPS, log_interval=LOG_INTERVAL, callback=callback)
    model.learn(total_timesteps=NB_TRAIN_EPS * EP_TIMESTEPS, log_interval=LOG_INTERVAL)
    model.save(os.path.join(logdir, algo))

    plot_results(logdir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', type=str, default=ENV, help='the name of the OpenAI Gym environment that you want to train on')
    parser.add_argument('--trial_id', type=int, default='0', help='trial identifier, name of the saving folder')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1e6), help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--algo', type=str, default=ALGO, help='underlying learning algorithm: td3 or ddpg')
    kwargs = vars(parser.parse_args())
    launch(**kwargs)