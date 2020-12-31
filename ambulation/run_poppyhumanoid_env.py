import gym
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('PoppyHumanoid-v0')
env = DummyVecEnv([lambda : env])
env.render()
env.reset()

while True:
    env.step(env.action_space.sample())
    env.render()
