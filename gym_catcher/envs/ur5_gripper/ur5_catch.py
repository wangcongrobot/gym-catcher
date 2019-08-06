import os
# from gym import utils
# from gym.envs.robotics import fetch_env


# # Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5_gripper', 'ur5gripper.xml')
print(MODEL_XML_PATH)

# class FetchCatchEnv(fetch_env.FetchEnv, utils.EzPickle):
#     def __init__(self, reward_type='sparse'):
#         initial_qpos = {
#             'robot0:slide0': 0.4049,
#             'robot0:slide1': 0.48,
#             'robot0:slide2': 0.0,
#         }
#         fetch_env.FetchEnv.__init__(
#             self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
#             gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
#             obj_range=0.15, target_range=0.15, distance_threshold=0.05,
#             initial_qpos=initial_qpos, reward_type=reward_type)
#         utils.EzPickle.__init__(self)


from gym import utils
from gym_catcher.envs import ur5_gripper_env


class UR5CatchEnv(ur5_gripper_env.UR5GripperEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse',
                 add_high_res_output=False, no_movement=False, stack_frames=False, camera_3=False):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            # 'shoulder_pan_joint': 1.57, 
            # 'shoulder_lift_joint': 1.57, 
            # 'elbow_joint': 1.57, 
            # 'wrist_1_joint': -1.57, 
            # 'wrist_2_joint': 1.57, 
            # 'wrist_3_joint': 1.57
        }
        ur5_gripper_env.UR5GripperEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.3, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            add_high_res_output=add_high_res_output, 
            no_movement=no_movement,
            stack_frames=stack_frames, 
            camera_3=camera_3
        )
        utils.EzPickle.__init__(self)

# class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
#     def __init__(self, reward_type='sparse'):
#         initial_qpos = {
#             'robot0:slide0': 0.405,
#             'robot0:slide1': 0.48,
#             'robot0:slide2': 0.0,
#             'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
#         }
#         fetch_env.FetchEnv.__init__(
#             self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
#             gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
#             obj_range=0.15, target_range=0.15, distance_threshold=0.05,
#             initial_qpos=initial_qpos, reward_type=reward_type)
#         utils.EzPickle.__init__(self)
