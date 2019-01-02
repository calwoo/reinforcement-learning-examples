"""
I'm pretty sure that if I just copy and paste the entire thing code from frozen-lake.py and just change the
environment this will work pretty flawlessly...
"""
import numpy as np 
import gym 
import random

# Initialize environment and hyperparameters
env = gym.make("Taxi-v2")
action_size = env.action_space.n
state_size = env.observation_space.n 
# The Q-table is literally just a lookup table (our MDP is finite-state finite-action) over all
# states and all actions
qtable = np.zeros((state_size, action_size))
epochs = 20000
discount_factor = 0.8
num_of_steps = 100
learning_rate = 0.9

# Our policy will be epsilon-greedy, i.e. we will pick a uniformly random number and compare it to a
# progressively shrinking epsilon, which dictates whether we take a greedy policy or a random-choice policy.
epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay_rate = 0.005

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
        # Use the Q-learning update rule to update the lookup table. Theoretically, the bellman equation
        # provides the update rule, which is effectively a gradient descent with the "bellman loss function".
        max_next_qvalues = np.max(qtable[new_state,:])
        update = reward + discount_factor * max_next_qvalues - qtable[state, action]
        qtable[state, action] += learning_rate * update
        # Update episodic parameters
        total_rewards += reward
        state = new_state
        # Finish episode if done flag raised
        if done:
            break
    
    # After episode is over, lower epsilon a bit as lookup table gets better and better and we can rely on
    # exploiting the greedy policy more.
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

"""
After we've trained our model using the Q-learning algorithm, we use the learned lookup table to play the
taxi game. We'll visualize this game using gym's render tool.
"""
env.reset()
for episode in range(3):
    state = env.reset()
    done = False
    print("Episode %d:" % episode)

    for step in range(num_of_steps):
        action = np.argmax(qtable[state,:])
        new_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            print("number of steps: %d" % step)
            break
        state = new_state

# Finally, when all is done, close the environment.
env.close()
