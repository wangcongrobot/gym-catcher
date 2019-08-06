# gym env to catch a moving object using ur5 in mujoco simulator

cd gym-catcher
pip install -e .

'''python
import gym
import gym_catcher

env = gym.make('UR5Catch-v0')
env.render()
'''

or:
python gym_catcher/test/test_ur5.py
