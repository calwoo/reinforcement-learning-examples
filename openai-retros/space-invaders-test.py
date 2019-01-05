"""
We're gonna test out the OpenAI retro library!
"""
import numpy as np 
import gym
import retro
import random

env = retro.make(game="SpaceInvaders-Atari2600")
action_size = env.action_space.n
state_space = env.observation_space
print(action_size)

frame = env.reset()
while True:
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    print(action, reward)
    env.render()
    if done:
        print("episode end!")
        break
env.close()