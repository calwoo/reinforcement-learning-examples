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

# Our policy will be epsilon-greedy, i.e. we will pick a uniformly random number and compare it to a
# progressively shrinking epsilon, which dictates whether we take a greedy policy or a random-choice policy.
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.05

"""
Now we implement the Q-learning algorithm. As the Q-table is already initialized, we proceed in 3 steps:
1) Select epsilon-greedy action using current Q-table estimates.
2) Proceed to next state using chosen action and collect reward.
3) Update Q-table using the Bellman equation update rule. Repeat.
"""
rewards = []
for episode in range(epochs):
    state = env.reset()
    done = False
    total_rewards = 0

    for step in range(num_of_steps):
        # Select epsilon-greedy action
        prob = random.uniform(0, 1)
        if prob > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        # Move to new state with action and collect reward
        new_state, reward, done, info = env.step(action)
        """
        The things that env.step(action) returns is:
        a) The observation, or new state
        b) Reward for performing action in current state
        c) A done (boolean) flag to tell us if we reached end goal or not
        d) Diagnostic data. We usually ignore this, unless for debugging.
        """
        # Use the Q-learning update rule to update the lookup table. Theoretically, the bellman equation
        # provides the update rule, which is effectively a gradient descent with the "bellman loss function".
        next_qvalues = []
        