import numpy as np

from gym.envs.robotics import rotations, utils
from gym_catcher.envs import robot_env

from gym_catcher.utils.dm_utils.rewards import tolerance 

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5PickEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
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

        # add some variables
        self.action = None
        self.end_location = None
        self.n_actions = 7

        super(UR5PickEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # add new function for catch env
    def _restart_target(self):
        target_x = self.np_random.uniform(low=0.52, high=0.84)
        target_y = self.np_random.uniform(low=-0.23, high=0.23)

        self.end_location = [2.2 - target_x, 0.75 + target_y, 0.66]

        y = self.np_random.uniform(low=-0.5, high=0.5)
        z = self.np_random.uniform(low=-0.15, high=0.)
        z_offset = 0.34 + z
        v_z = self.np_random.uniform(low=1.9, high=2.3)

        del_y = y - target_y

        target_dist = np.sqrt(del_y * del_y + target_x * target_x)

        sin_theta = target_x / target_dist

        v = target_dist * 9.81 / (v_z + np.sqrt(v_z * v_z + 19.62 * z_offset))

        v_x = v * sin_theta
        v_y = np.sign(del_y) * v * np.sqrt(1. - sin_theta * sin_theta)

        self.sim.data.set_joint_qpos('tar:x', 0.0)
        self.sim.data.set_joint_qpos('tar:y', y)
        self.sim.data.set_joint_qpos('tar:z', z)
        self.sim.data.set_joint_qvel('tar:x', - v_x)
        self.sim.data.set_joint_qvel('tar:y', - v_y)
        self.sim.data.set_joint_qvel('tar:z', v_z)


    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        # print("_set_action:", action)
        assert action.shape == (7,)
        self.action = action
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0.707, -0.707, 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _set_action_fetch(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
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
            object_pos = self.sim.data.get_site_xpos('object0') 
            # rotations object_rot - Yes. That is the orientation of the object with respect to world frame.
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities object_velp - Positional velocity of the object with respect to the world frame
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            # object_velr - Rotational velocity of the object with respect to the world frame
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
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
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self._restart_target()
        self.sim.forward()

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

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
        print("gripper_target:", gripper_target)
        print("currrent gripper position:", self.sim.data.get_site_xpos('gripperpalm'))
        gripper_rotation = np.array([1., 0., 1., 0.]) # fixed oritation to grasp
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripperpalm').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(UR5PickEnv, self).render(mode, width, height)
