from metaworld.benchmarks import MT10
from gym.envs.metaworld.base import MTEnv
import gym
def MT10HelperEnv():
    return MTEnv(MT10.get_train_tasks())
