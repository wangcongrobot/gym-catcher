from gym.envs.registration import register


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='UR5Gripper{}-v0'.format(suffix),
        entry_point='gym_catcher.envs:UR5GripperEnv',
        max_episode_steps=250,
        kwargs={**kwargs, **dict(
            add_high_res_output=False,
            no_movement=False,
            stack_frames=False,
            camera_3=False
        )}
    )

    register(
        id='UR5PickAndPlace{}-v0'.format(suffix),
        entry_point='gym_catcher.envs:UR5PickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='UR5Catch{}-v0'.format(suffix),
        entry_point='gym_catcher.envs:UR5CatchEnv',
        kwargs=kwargs,
        max_episode_steps=50
    )