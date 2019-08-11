import os

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5_gripper', 'ur5gripper.xml')
print(MODEL_XML_PATH)

from gym import utils
from gym_catcher.envs import ur5_catch_env


class UR5CatchEnv(ur5_catch_env.UR5CatchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse',
                 add_high_res_output=False, no_movement=False, stack_frames=False, camera_3=False):
        initial_qpos = {
            # the initial position of the robot base: x, y, z
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.40,
            # the pose of the object: x, y, z, w, x, y, z
            'object0:joint': [1.35, 0.53, 0.4, 1., 0., 0., 0.],
        }
        ur5_catch_env.UR5CatchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.3, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)