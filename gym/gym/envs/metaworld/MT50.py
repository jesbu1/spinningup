from metaworld.benchmarks import MT50
from gym.envs.metaworld.base import MTEnv
import gym
def MT50HelperEnv():
    return MTEnv(MT50.get_train_tasks())
