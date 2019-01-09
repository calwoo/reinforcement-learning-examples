import numpy as np 
import gym
import tensorflow as tf 
import random

env = gym.make("MountainCar-v0")
env._max_episode_steps = 1000

# hyperparameters
state_size = 2
action_size = env.action_space.n
num_episodes = 10000
learning_rate = 0.05
discount_factor = 0.99

# discount rewards helper function
def discount_rwds(rewards):
    discounted_rwds = np.zeros_like(rewards)
    """
    We want to keep track of all cumulative timestepped rewards. For example, the expected discounted rewards
    at time step t is given by sum_k gamma^k * r_{t+k} where sum goes on until end of episode.
    """
    total_rewards = 0
    for t in range(len(discounted_rwds))[::-1]:
        r_t = rewards[t]
        total_rewards = r_t + discount_factor * total_rewards
        discounted_rwds[t] = total_rewards
    # Normalize discounted rewards
    mean = np.mean(discounted_rwds)
    std = np.std(discounted_rwds)
    normalized_discounted_rwds = (discounted_rwds - mean) / std
    return normalized_discounted_rwds

"""
Our policy will be given by a neural network pi_theta, which will take in a state and spit out a probability
distribution over the possible actions at that state.
"""
class Policy:
    def __init__(self, state_size, action_size, learning_rate):
        self.inputs = tf.placeholder(tf.float32, [None, state_size])
        self.actions = tf.placeholder(tf.float32, [None, action_size])
        self.discounted_rwds = tf.placeholder(tf.float32, [None,])
        """
        Our neural net will be a 4-layer neural net with 3 fully-connected layers with a softmax output.
        The score function is the expected discounted rewards that we wish to maximize.
        """
        self.fc1 = tf.layers.dense(
            inputs=self.inputs,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.fc2 = tf.layers.dense(
            inputs=self.fc1,
            units=action_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.fc3 = tf.layers.dense(
            inputs=self.fc2,
            units=action_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.outputs = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.fc3,
            labels=self.actions
        )
        self.action_dist = tf.nn.softmax(self.fc3)
        # Loss function and optimizer
        self.loss = tf.reduce_mean(self.outputs * self.discounted_rwds)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

policy = Policy(state_size, action_size, learning_rate)

"""
Now we get to train our policy.
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

max_reward_so_far = 0

for episode in range(num_episodes):
    state = env.reset()
    env.render()
    states = []
    rewards = []
    actions = []
    while True:
        action_dist = sess.run(
            policy.action_dist,
            feed_dict={policy.inputs: state.reshape([1,state_size])}
        )
        action = random.choices(
            range(action_dist.shape[1]),
            weights=action_dist.ravel()
        )[0]
        
        next_state, reward, done, _ = env.step(action)
        env.render()

        # collect your data
        states.append(state)
        rewards.append(reward)
        # one hot the action and then append to actions
        one_hot_action = np.zeros(action_size)
        one_hot_action[action] = 1
        actions.append(one_hot_action)

        if done:
            # finally do the policy update
            total_rewards = np.sum(rewards)
            if total_rewards > max_reward_so_far:
                max_reward_so_far = total_rewards
            print("episode %d: reward = %.3f / max so far = %.3f" % (episode, total_rewards, max_reward_so_far))
            discounted_rwds = discount_rwds(rewards)
            loss, _ = sess.run(
                [policy.loss, policy.optimizer],
                feed_dict={
                    policy.inputs: np.vstack(np.array(states)),
                    policy.actions: np.vstack(np.array(actions)),
                    policy.discounted_rwds: discounted_rwds
                }
            )
            break

        state = next_state