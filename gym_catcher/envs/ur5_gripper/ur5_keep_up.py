import os

# # Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5_gripper', 'ur5_keep_up.xml')
# MODEL_XML_PATH = '/home/cong/workspace/DHER/gym-catcher/gym_catcher/envs/asset/ur5_gripper/ur5gripper.xml'
print(MODEL_XML_PATH)

from gym import utils
from gym_catcher.envs import ur5_keep_up_env

class UR5KeepUpEnv(ur5_keep_up_env.UR5KeepUpEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.,
            'robot0:slide1': 0.,
            'robot0:slide2': 0.0,
            # 'object0:joint': [1.35, 0.53, 0.4, 1., 0., 0., 0.],
            # observation space
            # 'robot0:slide0':,
            # 'robot0:slide1':,
            # 'robot0:slide2',
            'shoulder_pan_joint': 0.,
            'shoulder_lift_joint': 0.,
            'elbow_joint': 0.,
            'wrist_1_joint': 0.,
            'wrist_2_joint': 0.,
            'wrist_3_joint': 0.,
            # 'gripperpalm_finger_1_joint',
            # 'gripperfinger_1_joint_1',
            # 'gripperfinger_1_joint_2',
            # 'gripperfinger_1_joint_3',
            # 'gripperpalm_finger_2_joint',
            # 'gripperfinger_2_joint_1',
            # 'gripperfinger_2_joint_2',
            # 'gripperfinger_2_joint_3',
            # 'gripperpalm_finger_middle_joint',
            # 'gripperfinger_middle_joint_1',
            # 'gripperfinger_middle_joint_2',
            # 'gripperfinger_middle_joint_3',
            # 'tar:joint'
        }
        ur5_keep_up_env.UR5KeepUpEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.5, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
