import gym
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('PoppyStandup-v0')
# env = gym.make('PoppyKeepStanding-v0')
env = DummyVecEnv([lambda : env])
env.render()
env.reset()

while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
