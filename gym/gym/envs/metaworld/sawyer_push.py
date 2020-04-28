from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv
from gym.envs.metaworld.base import MTEnv
import gym
import numpy as np
from gym import spaces
def SawyerPushEnv():
    env = SawyerReachPushPickPlaceEnv(task_type='push', obs_type='with_goal', random_init=True)
    # Only works if random_init = true
    #env.goal_low = (0.1, -0.2, 0.85)
    #env.goal_high = (0.1, 0.2, 0.95)
    #env.obj_low = (0.1, -0.4, 0.2)
    #env.obj_high = (0.1, 0.4, 1.0)
    #env.obj_and_goal_space = spaces.Box(
    #    np.hstack((env.obj_low, env.goal_low)),
    #    np.hstack((env.obj_high, env.goal_high)),
    #)
    return env
