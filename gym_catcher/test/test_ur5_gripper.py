import gym

import matplotlib.pyplot as plt

# import the new env
import gym_catcher 

mode = 'human'
#mode = 'rgb_array'

env = gym.make("UR5Gripper-v0")

env.render('human')
#env = gym.wrappers.Monitor(env, './video', force=True)
#plt.imshow(env.render(mode='rgb_array', camera_id=-1))
#plt.show()
for i in range(20):
  env.reset()
  env.render('human')
  for i in range(20):
    action = env.action_space.sample()
    action = [0, 0, 0, 0, 0, 0, 0]
    print("action_space:", env.action_space)
    print("action space sample:", action)
    obs, reward, done, info = env.step(action)
    print("observation:", obs)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
    env.render('human')

