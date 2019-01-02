"""
For games with very complex action spaces, setting up a Q-table to store the Q-values is pretty bad. The size
of it drains memory requirements quickly, especially since the preferred actions leave the lookup table very
sparse. To attempt to rectify this, we start working on learning the Q-values by storing it as a massive
function-- a neural network.

This forms the basis of the DQN (Deep Q-learning network).
"""
import numpy as np 
import gym
import random
import tensorflow as tf
from vizdoom import *
import sklearn
from collections import deque

"""
A deep Q-learning network operates the same as the traditional Q-learning algorithm (uses the same update rule),
where the difference is that the lookup table is replaced by a trained neural network that takes as input states
and approximates the Q-values for each of the actions based on that state.
"""
# First we gotta get Doom working.
# The vizdoom library allows us to get a game working with configuration and scenario files.
def make_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    # Possible actions our MDP can take
    