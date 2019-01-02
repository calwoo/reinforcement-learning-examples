"""
We will implement a Q-learning algorithm to play the frozen-lake game environment
given in OpenAI Gym. Apparently, frozen-lake is a game where you try to traverse to the goal
through slippery ice (which introduces a stochastic element). However, I know nothing else
about this game.
"""
import numpy as np 
import gym 
import random

# Initialize environment and hyperparameters
env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n 
# The Q-table is literally just a lookup table (our MDP is finite-state finite-action) over all
# states and all actions
qtable = np.zeros((state_size, action_size))
epochs = 2000
discount_factor = 0.8
num_of_steps = 100
learning_rate = 0.6

# Our policy will be epsilon-greedy, i.e. we will 