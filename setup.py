from setuptools import setup

setup(
    name='gym_catcher',
    version='0.0.1',
    description='The dynamic goal gym using ur5 and 3 finger gripper.',
    author='Cong Wang',
    author_email='wangcongrobot@gmail.com',
    install_requires=['gym==0.14.0', 'numpy', 'mujoco_py==2.0.2.2'])
