import gym

import matplotlib.pyplot as plt

# import the new env
import gym_catcher 

mode = 'human'
#mode = 'rgb_array'

env = gym.make("UR5KeepUp-v0")
num_actuator = env.sim.model.nu
print('num_actuator: ', num_actuator)
env.render('human')
#env = gym.wrappers.Monitor(env, './video', force=True)
#plt.imshow(env.render(mode='rgb_array', camera_id=-1))
#plt.show()
for i in range(20):
  env.reset()
  env.render('human')
  for i in range(50):
    action = env.action_space.sample()
    #action = [0, 0.5, 0.5, 0.5, 0.5, 0, 0]
    print("action_space:", env.action_space)
    print("action space sample:", action)
    obs, reward, done, info = env.step(action)
    print("observation:", obs)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
    env.render('human')
    print("number actuator: ", num_actuator)
    print("name: ", env.sim.model.name_actuatoradr)
    print("actuator contrl range: ", env.sim.model.actuator_ctrlrange)
