# Change Fetch robot environment to UR5 with gripper to catch object
# 
import numpy as np

from gym.envs.robotics import rotations, utils
from gym_catcher.envs import robot_env_ur5

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # print("goal_a: ", goal_a)
    # print("goal_b: ", goal_b)
    # print("shape: ", goal_a.shape)
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5KeepUpEnv(robot_env_ur5.RobotEnv):
    """
    Superclass for all UR5 environments.
            derive from Fetch env    
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        # add some variables for keep up
        self.initial_h = 2.2
        self.sqrt_2_g = 4.429446918
        self.target_z_l = 4

        self.action = None
        self.end_location = None
        # action dimentions
        self.n_actions = 7
        # calculate the time from one epoch starting
        # self.counts_from_start = 0

        super(UR5KeepUpEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------



    def compute_reward_catch1(self, achieved_goal, desired_goal, info):
        '''
        The reward function includes three parts: 
        1. shaping reward: distance of the gripper and the object
        2. sparse reward: if success, then get a sparse reward
        3. time reward: panalty when using more time
        '''
        
        print("self.action:", self.action)
        reward_ctrl = - 0.05 * np.square(self.action[:3]).sum()
        print("reward_ctrl", reward_ctrl)
        dist_to_end_location = np.linalg.norm(self.sim.data.get_site_xpos('gripperpalm') - 
                                              self.end_location)
        print("gripperpalm: ", self.sim.data.get_site_xpos('gripperpalm'))
        print("end_location: ", self.end_location)
        print("dist_to_end_location: ", dist_to_end_location)
        reward_dist = tolerance(dist_to_end_location, margin=0.8, bounds=(0., 0.02),
                                sigmoid='linear', 
                                value_at_margin=0.)

        print("reward_dist:", reward_dist)
        reward = 0.25 * reward_dist
        
        # if z < 0.1, then restart
        if self.sim.data.get_site_xpos('object')[2] < 0.1:
            self._restart_target()
        
        sparse_reward = 0.
        dist = np.linalg.norm(self.sim.data.get_site_xpos('gripperpalm') - # the position of the end-effector
                              self.sim.data.get_site_xpos('object')) # the position of target
        print("dist:", dist)
        if dist < 0.05:
            reward += 2.
            sparse_reward += 1.
            self._restart_target()

        reward += reward_ctrl

        # reward_time = -0.01 * self.counts_in_epoch
        print("counts in epoch: ", self.counts_in_epoch)

        # reward += reward_time

        info = dict(scoring_reward=sparse_reward)
        print("sparse_reward: ", sparse_reward)
        print("reward_ctrl: ", reward_ctrl)
        # print("reward_time: ", reward_time)
        print("total reward:", reward)
        
        # reward = np.random.random_sample()
        return reward

    # compute reward keep up
    def compute_reward111(self, achieved_goal, desired_goal, info):

        obj_pos = self.sim.data.get_site_xpos('object')
        palm_pos = self.sim.data.get_mocap_pos('robot0:mocap')
        target_pos = self.sim.data.get_site_xpos('target')

        reward_ctrl = - 0.05 * np.square(self.action).sum()
        
        # dist_end_to_target1 = np.linalg.norm(self.sim.data.get_site_xpos('gripperpalm') - 
        #                                     self.sim.data.get_site_xpos('object'))
        # # print("dist1: ", dist_end_to_target1)
        # dist_end_to_target = np.linalg.norm(self.sim.data.get_mocap_pos('robot0:mocap') - 
        #                                     self.sim.data.get_site_xpos('object'))
        # # print("dist2: ", dist_end_to_target)
        # # print("gripperpalm: ", self.sim.data.get_site_xpos('gripperpalm'))
        # # print("robot0:mocap: ", self.sim.data.get_mocap_pos('robot0:mocap'))
        # # print("tar: ", self.sim.data.get_site_xpos('object'))
        dist_finger_1_to_target = np.linalg.norm(self.sim.data.get_site_xpos('gripperfinger_1_polp_3') - 
                                                 self.sim.data.get_site_xpos('object'))
        dist_finger_2_to_target = np.linalg.norm(self.sim.data.get_site_xpos('gripperfinger_2_polp_3') - 
                                                 self.sim.data.get_site_xpos('object'))
        dist_finger_middle_to_target = np.linalg.norm(self.sim.data.get_site_xpos('gripperfinger_middle_polp_3') - 
                                                      self.sim.data.get_site_xpos('object'))

        # reward_dist = tolerance(dist_end_to_target, margin=0.5, bounds=(0., 0.02),
        #                         sigmoid='linear',
        #                         value_at_margin=0.)
        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        if (dist_finger_1_to_target < 0.05) or (dist_finger_2_to_target < 0.05) or (dist_finger_middle_to_target < 0.05): # if grap the object
            reward += 1.0   # bonus for grap the object
            reward += -0.5 * np.linalg.norm(palm_pos - target_pos) # make hand go to target
            reward += -0.5 * np.linalg.norm(obj_pos - target_pos) # make object go to target
        if np.linalg.norm(obj_pos-target_pos) < 0.1:
            reward += 10.0  # bonus for object close to target
        if np.linalg.norm(obj_pos-target_pos) < 0.05:
            reward += 20.0  # bonus for object "very" close to target
        reward += reward_ctrl


        # sparse_reward = 0.
        # if dist_end_to_target < 0.05:
        #     sparse_reward += 1.
        #     if (dist_finger_1_to_target < 0.05) or (dist_finger_2_to_target < 0.05) or (dist_finger_middle_to_target < 0.05):
        #         sparse_reward += 2.
        

        # reward = 0.2 * reward_dist + reward_ctrl
        # print("reward_ctrl: ", reward_ctrl)
        # print("dist_end_to_target: ", dist_end_to_target)
        # print("reward_dist: ", reward_dist)
        # print("sparse_reward: ", sparse_reward)
        # reward = reward_ctrl + dist_end_to_target * -0.5 + sparse_reward

        # done = False
        # if self.sim.data.get_site_xpos('object')[2] < 0.1: # z < 0.1
        #     done = True
        #     reward -1.

        # sparse_reward = 0.
        # if dist < 0.05:
        #     reward +=2.
        #     sparse_reward += 1.
        


        # info = dict(scoring_reward=sparse_reward)

        # return reward, done, info
        return reward
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # compute sparse rewards
        # self._check_success()
        # reward 

        # add in shaped rewards
        # if self.reward_shaping:
            # staged_rewards = self.staged_rewards()
            # reward += max(staged_rewards)
        staged_rewards = self.staged_rewards()
        print("one step reward: ", np.sum(staged_rewards))
        return np.sum(staged_rewards)
    
    def staged_rewards(self):
        """
        Returns staged rewawrds based on current physical states.
        Stages consist of following, reaching, grasping, lifting, and hovering.
        """
        # importance degree of four shaping reward
        control_mult = -0.05
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # control reward
        reward_ctrl = np.square(self.action).sum() * control_mult
        print("reward_ctrl: ", reward_ctrl)

        # following reward

        # reaching reward
        obj_pos = self.sim.data.get_site_xpos('object')
        palm_pos = self.sim.data.get_mocap_pos('robot0:mocap')
        target_pos = self.sim.data.get_site_xpos('target')
        dist = np.linalg.norm(palm_pos - obj_pos)
        # the larger distance, the fewer reward
        # convert the dist to range (0, 1)
        reward_reach = (1 - np.tanh(10.0 * dist)) * reach_mult
        print("reward_reach: ", reward_reach)

        # grasping reward
        # int(False) = 0 int(True) = 1
        reward_grasp = int(self._check_grasp()) * grasp_mult
        print("reward_grasp: ", reward_grasp)

        # lifting reward


    def _check_grasp(self):
        """
        Return True if the gripper has grasped the object.
        Using the contact detection between gripper and object.
        First, palm contact must be true.
        Second, three (or two) fingers contact with the object can return True.
        """
        # get the contact geom information from ur5_keep_up.xml file
        finger_1_geom_names = ["f1_l0", "f1_l1", "f1_l2", "f1_l3"]
        finger_2_geom_names = ["f2_l0", "f2_l1", "f2_l2", "f2_l3"]
        finger_3_geom_names = ["f3_l0", "f3_l1", "f3_l2", "f3_l3"]
        palm_geom_name = "gripperpalm"
        object_geom_name = "object"

        # get the geom ids from the names
        finger_1_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_1_geom_names
        ]
        finger_2_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_2_geom_names
        ]
        finger_3_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_3_geom_names
        ]
        palm_geom_id = self.sim.model.geom_name2id(palm_geom_name)
        object_geom_id = self.sim.model.geom_name2id(object_geom_name)

        # touch/contact detection flag
        touch_finger_1 = False
        touch_finger_2 = False
        touch_finger_3 = False
        touch_palm = False
        has_grasp = False

        # int ncon: number of detected contacts
        for i in range(self.sim.data.ncon):
            # list of all detected contacts
            contact = self.sim.data.contact[i]
            # if the object is in the detected contact geom1
            if contact.geom1 in object_geom_id:
                # wether the finger 1 in the detected contacts
                if contact.geom2 in finger_1_geom_ids:
                    # result: finger 1 touched the object
                    touch_finger_1 = True
                if contact.geom2 in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
                if contact.geom2 in finger_3_geom_ids:
                    # finger 3 touched the object
                    touch_finger_3 = True
                if contact.geom2 in palm_geom_id:
                    # palm touched the object
                    touch_palm = True
            # if the object is in the detected contact geom2
            elif contact.geom2 in object_geom_id:
                if contact.geom1 in finger_1_geom_ids:
                    # finger 1 touched the object
                    touch_finger_1 = True
                if contact.geom1 in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
                if contact.geom1 in finger_3_geom_ids:
                    # finger 3 touched the object
                    touch_finger_3 = True
                if contact.geom1 in palm_geom_id:
                    # palm touched the object
                    touch_palm = True

        # TODO: two finger may also has grasp
        # the palm touch must be true first
        if touch_palm:
            # has three finger touch must be true
            if touch_finger_1 and touch_finger_2 and touch_finger_3:
                has_grasp = True
            # has two finger touch maybe true
            elif touch_finger_1 and touch_finger_3:
                has_grasp = True
            # has two finger touch maybe true
            elif touch_finger_2 and touch_finger_3:
                has_grasp = True
        if has_grasp:
            print("Get a successful grasp!")

        return has_grasp


    def _check_success(self):
        """
        Return True if task has been completed.
        """

        pass

    # new reward compute function for catch env
    def compute_reward_catch(self, achieved_goal, goal, info):
        print("compute_reward")
        print("self.action:",  self.action)
        reward_ctrl = - 0.05 * np.square(self.action).sum()

        dist_to_end_location = np.linalg.norm(self.sim.data.get_site_xpos('gripperpalm') -
                                              self.end_location)
        reward_dist = tolerance(dist_to_end_location, margin=0.8, bounds=(0., 0.02),
                                sigmoid='linear',
                                value_at_margin=0.)

        reward = 0.25 * reward_dist

        if self.sim.data.get_site_xpos('object')[2] < 0.1: # if z < 0.1, then drop out and restart
            self._restart_target()

        sparse_reward = 0.
        dist = np.linalg.norm(self.sim.data.get_site_xpos('gripperpalm') -
                              self.sim.data.get_site_xpos('object'))
        if dist < 0.05:
            reward += 20.
            sparse_reward += 10.
            self._restart_target()

        reward += reward_ctrl

        info = dict(scoring_reward=sparse_reward)

        return reward, False, info

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()
            # self.counts_from_start += 1
            # print("counts_from_start: ", self.counts_from_start)

    def _set_action(self, action):
        # print("_set_action:", action)
        assert action.shape == (self.n_actions,)
        self.action = action
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        # grip_pos - Position of the gripper given in 3 positional elements and 4 rotational elements
        grip_pos = self.sim.data.get_site_xpos('gripperpalm')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp - The velocity of gripper moving
        grip_velp = self.sim.data.get_site_xvelp('gripperpalm') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            # object_pos - Position of the object with respect to the world frame
            object_pos = self.sim.data.get_site_xpos('object') 
            # rotations object_rot - Yes. That is the orientation of the object with respect to world frame.
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object'))
            # velocities object_velp - Positional velocity of the object with respect to the world frame
            object_velp = self.sim.data.get_site_xvelp('object') * dt
            # object_velr - Rotational velocity of the object with respect to the world frame
            object_velr = self.sim.data.get_site_xvelr('object') * dt
            # gripper state
            # object_rel_pos - Position of the object relative to the gripper
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        # gripper_state - No. It's not 0/1 signal, it is a float value and varies from 0 to 0.2 
        # for fetch robot. This varied gripper state helps in grasping different sized 
        # object with different strengths.
        # gripper_state - The quantity to measure the opening of gripper
        gripper_state = robot_qpos[-2:]
        # gripper_vel - The velocity of gripper opening/closing
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('gripperpalm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('object')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # self._restart_target()

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) \
                         + self.sim.data.get_site_xpos('gripperpalm')
        # gripper_target = [-0.6, -0.5, 0.8]
        print("gripper_target:", gripper_target)
        print("currrent gripper position:", self.sim.data.get_site_xpos('gripperpalm'))
        gripper_rotation = np.array([0., 0., 1., 0.]) # fixed oritation to grasp
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripperpalm').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object')[2]

    def render(self, mode='human', width=500, height=500):
        return super(UR5KeepUpEnv, self).render(mode, width, height)
