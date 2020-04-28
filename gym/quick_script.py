import gym
import numpy as np
env = gym.make('SawyerPush-v0')
for _ in range(100):
    env.reset()
    for i in range(150):
        env.render()
        env.step(np.random.uniform(0, 1, size=(4,)))
