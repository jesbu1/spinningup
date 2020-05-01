from metaworld.benchmarks import MT40
from gym.envs.metaworld.base import MTEnv
import gym
def MT40HelperEnv():
    return MTEnv(MT40.get_train_tasks())
