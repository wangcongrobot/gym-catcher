import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class RobotEnvM(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        # add function
        #self.renderer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        # self.observation_space = spaces.Dict(dict(
        #     desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     image=spaces.Box(-np.inf, np.inf, shape=obs['image'].shape, dtype='float32'),
        #     vector=spaces.Box(-np.inf, np.inf, shape=obs['vector'].shape, dtype='float32')
        # ))
        # change observation space
        # self.observation_space = {}
        # for key, value in obs.items():
        #     self.observation_space[key] = spaces.Box(-np.inf, np.inf,
        #                                              shape=value.shape,
        #                                              dtype=value.dtype) 
        # fetch catch env   
        # self.observation_space = spaces.Dict({
        #     'image': spaces.Box(low=0, high=255, shape=(297, 528, 3), dtype=np.uint8),
        #     'vector': spaces.Box(spaces.Box(-np.inf, np.inf, shape=(3,3))),
        #     # 'vector': spaces.Dict(spaces.Box(-np.inf, np.inf, shape=(3,)
        #         # spaces.Box(-np.inf, np.inf, shape=(3,)),
        #         # spaces.Box(-np.inf, np.inf, shape=(3,)),
        #         # spaces.Box(-np.inf, np.inf, shape=(3,))),
        #     'achieved_goal': spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     'desired_goal': spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
        # })

#         self.observation_space = spaces.Dict(dict(x=spaces.Box(-high[0], high[0]), 
#                                             fx=spaces.Box(-high[1], high[1]), 
#                                             theta=spaces.Box(-high[2], high[2]), 
#                                             ftheta=spaces.Box(-high[3], high[3])))
#     Example usage:
#     self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

#     Example usage [nested]:
#     self.nested_observation_space = spaces.Dict({
#         'sensors':  spaces.Dict({
#             'position': spaces.Box(low=-100, high=100, shape=(3,)),
#             'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
#             'front_cam': spaces.Tuple((
#                 spaces.Box(low=0, high=1, shape=(10, 10, 3)),
#                 spaces.Box(low=0, high=1, shape=(10, 10, 3))
#             )),
#             'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
#         }),
#         'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
#         'inner_state':spaces.Dict({
#             'charge': spaces.Discrete(100),
#             'system_checks': spaces.MultiBinary(10),
#             'job_status': spaces.Dict({
#                 'task': spaces.Discrete(5),
#                 'progress': spaces.Box(low=0, high=100, shape=()),
#             })
#         })
#     })
    
#     The observation space is defined as a single camera image from 
#     the front camera using the Box space from gym:

# "observation_space" : 
#   spaces.Box(
#       low=0,
#       high=255,
#       shape=(297, 528, 3),
#       dtype=np.uint8
#     ) # RGB image from front camera

# The shape tuple specifies the image size. The simulator API will always 
# send 1920x1080 images. If any other size is used to define the observation space,
#  the camera images will be resized to the specified size before being passed 
#  on as an observation.


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        
        # new reward function
        reward, done, info = self.compute_reward(None, None, {})

        obs = self._get_obs()

        # done = False
        # info = {
        #     'is_success': self._is_success(obs['achieved_goal'], self.goal),
        # }
        # reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnvM, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    # def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
    #     self._render_callback()
    #     if mode == 'rgb_array': # get image data
    #         self._get_viewer(mode).render(width, height)
    #         # window size used for old mujoco-py:
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=False)
    #         # original image is upside-down, so flip it
    #         return data[::-1, :, :]
    #     elif mode == 'human':
    #         self._get_viewer(mode).render()
    # add user defined camera name
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_id=-1):
        self._render_callback()
        if mode == 'rgb_array': # get image data
            # camera_id = None
            # if camera_name in self.model.camera_names:
                # camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height, camera_id)
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    # from mujoco_env
    # def render_from_mujoco_env(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
    #     if mode == 'rgb_array':
    #         camera_id = None
    #         camera_name = 'track'
    #         # if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
    #             # camera_id = self.model.camera_name2id(camera_name)
    #         self._get_viewer(mode).render(width, height, camera_id=camera_id)
    #         # window size used for old mujoco-py:
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=False)
    #         # original image is upside-down, so flip it
    #         return data[::-1, :, :]
    #     elif mode == 'depth_array':
    #         self._get_viewer(mode).render(width, height)
    #         # window size used for old mujoco-py:
    #         # Extract depth part of the read_pixels() tuple
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
    #         # original image is upside-down, so flip it
    #         return data[::-1, :]
    #     elif mode == 'human':
    #         self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array': # add depth mode, similar with mujoco_env
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _get_viewer_from_mujoco_env(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer




    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
