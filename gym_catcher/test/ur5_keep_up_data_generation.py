import gym
import numpy as np
import gym_catcher

"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

max_episode_steps = 100

def main():
    env = gym.make('UR5KeepUp-v0')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_ur5"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    # object_pos - Position of the object with respect to the world frame
    objectPos = lastObs['observation'][3:6]
    # object_rel_pos - Position of the object relative to the gripper
    # object_rel_pos = object_pos - grip_pos
    object_rel_pos = lastObs['observation'][6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    # first make the gripper go slightly above the object
    # z += 0.03
    object_oriented_goal[2] += 0.0 

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)

    # end-effector approach to the object
    while np.linalg.norm(object_oriented_goal) >= 0.05 and timeStep <= max_episode_steps:
        env.render()
        print("step1: let the end-effector approach to the object")
        print("distance:", np.linalg.norm(object_oriented_goal))
        action = [0, 0, 0, 0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.0

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        # action[len(action)-1] = 0.05 #open
        # open the gripper [0, 0, 0, 0]
        # close the gripper [0.5, 0.5, 0.5, 0.0]
        action[3:] = [0, 0, 0, 0] # open the gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
    # close the gripper
    while np.linalg.norm(object_rel_pos) >= 0.05 and timeStep <= max_episode_steps :
        env.render()
        print("step2: close the gripper")
        print("distance:", np.linalg.norm(object_rel_pos))
        action = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        # action[len(action)-1] = -0.005
        action[3:] = [0.8, 0.8, 0.8, 0.0] # close

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    # move to the target goal
    while np.linalg.norm(goal - objectPos) >= 0.05 and timeStep <= max_episode_steps :
        env.render()
        print("step3: move to the target goal")
        print("distace:", np.linalg.norm(goal-objectPos))
        action = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6
        # print("move to target:", goal)
        # print("action:", action)
        # print("objectPos:", objectPos)

        # action[len(action)-1] = -0.005
        action[3:] = [0.8, 0.8, 0.8, 0.0] # close

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    while True: #limit the number of timesteps in the episode to a fixed duration
        env.render()
        print("step4: time limit")
        action = [0, 0, 0, 0, 0, 0, 0]
        # action[len(action)-1] = -0.005 # keep the gripper closed
        action[3:] = [0.5, 0.5, 0.5, 0.0]

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= max_episode_steps: 
            print("timeout!!!!!!!")
            break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()
