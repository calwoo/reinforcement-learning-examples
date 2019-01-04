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

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())