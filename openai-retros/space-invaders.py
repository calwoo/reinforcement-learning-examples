import numpy as np 
import gym
import random
import tensorflow as tf
import retro
import skimage
from skimage.color import rgb2gray # Grayscaling from RGB
import time
from collections import deque

# Set up environment and preprocessing functions
env = retro.make(game="SpaceInvaders-Atari2600")
num_actions = env.action_space.n
actions = np.identity(num_actions, np.int).tolist()

# STORE HYPERPARAMETERS HERE
state_dims = [110,84,4]
frame_stack_size = 4 # as per the Nature article
num_episodes = 30
num_steps_max = 100
minibatch_size = 32
learning_rate = 0.0005
discount_factor = 0.95
max_epsilon = 1.0 # epsilon greedy parameters
min_epsilon = 0.01
epsilon_decay_rate = 0.0001
memory_size = 1000000 # memory size from Nature article for experience replay

# Preprocess frames (stack of 4, downsampled and resized, cropping)
def preprocess_frame(frame):
    gray_frame = rgb2gray(frame)
    cropped_frame = gray_frame[8:-12,4:-12]
    normalized_frame = cropped_frame / 255.0
    resized_frame = skimage.transform.resize(normalized_frame, [110,84])
    return resized_frame

frames_queue = deque([np.zeros((110,84), dtype=np.int) for _ in range(frame_stack_size)], 
    maxlen=frame_stack_size)
def create_state(frames_queue, current_frame, new_epi_flag=False):
    frame = preprocess_frame(current_frame)
    # If new episode, restart the frames_queue with the current frame
    if new_epi_flag:
        frames_queue = deque([np.zeros((110,84), dtype=np.int) for _ in range(frame_stack_size)], 
            maxlen=frame_stack_size)
        for _ in range(frame_stack_size):
            frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    else:
        frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    return state, frames_queue

# Set up DQN
class NatureDQN:
    def __init__(self, state_dims, num_actions, learning_rate):
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        """
        The network in the Nature 2015 Deepmind article was a convolutional neural network with an 84x84x4 input
        coming from the preprocessing function, followed by:
        - conv layer with 32 8x8 filters with stride 4 -> ReLU activation
        - conv layer with 64 3x3 filters with stride 1 -> ReLU activation
        - fc layer with 512 ReLU units
        - fc output layer with single output for each action
        """
        self.inputs = tf.placeholder(tf.float32, shape=[None, *self.state_dims])
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        # The target is given by the Bellman equation
        self.target_Q = tf.placeholder(tf.float32, shape=[None])
        # Start the convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputs,
            filters=32,
            kernel_size=[8,8],
            strides=[4,4],
            padding="valid",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        self.conv1_bn = tf.layers.batch_normalization(
            inputs=self.conv1,
            training=True,
            epsilon=1e-5
        )
        self.conv1_relu = tf.nn.relu(self.conv1_bn)
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1_relu,
            filters=64,
            kernel_size=[3,3],
            strides=[1,1],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        self.conv2_bn = tf.layers.batch_normalization(
            inputs = self.conv2,
            training=True,
            epsilon=1e-5
        )
        self.conv2_relu = tf.nn.relu(self.conv2_bn)
        self.flatten = tf.layers.flatten(self.conv2_relu)
        self.fc_layer = tf.layers.dense(
            inputs=self.flatten,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.outputs = tf.layers.dense(
            inputs=self.fc_layer,
            units=num_actions,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        # Loss and Q-value
        self.Q_vals = tf.reduce_sum(self.actions * self.outputs, axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_vals))
        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

    def add_to_memory(self, experience):
        self.memory.append(experience)
    
    def sample(self, minibatch_size):
        current_memory_size = len(self.memory)
        indices = np.random.choice(np.arange(current_memory_size),
            size=minibatch_size,
            replace=False)
        minibatch = [self.memory[i] for i in indices]
        return minibatch
    
    def initialize_memory(self, env, actions, frames_queue):
        # Clear memory and start over
        self.memory.clear()
        # Create initial batch of experiences via random actions
        for i in range(minibatch_size):
            if i == 0:
                frame = env.reset()
                state, frames_queue = create_state(frames_queue, frame, True)
            action = random.choice(actions)
            next_frame, reward, done, _ = env.step(action)
            next_state, frames_queue = create_state(frames_queue, next_frame, False)
            if done:
                next_state = np.zeros(state.shape)
                self.add_to_memory((state, action, reward, next_state, done))
                frame = env.reset()
                state, frames_queue = create_state(frames_queue, frame, True)
            else:
                self.add_to_memory((state, action, reward, next_state, done))
                state = next_state
        return frames_queue

# Now we train the network
model = NatureDQN(state_dims, num_actions, learning_rate)
memory = Memory(memory_size)
frames_queue = memory.initialize_memory(env, actions, frames_queue)
sess = tf.Session()
checkpoint = tf.train.Saver()

sess.run(tf.global_variables_initializer())
env.reset()

training_flag = True

if training_flag:
    for episode in range(num_episodes):
        rewards = []
        decay_counter = 0
        frame = env.reset()
        state, frames_queue = create_state(frames_queue, frame, True)
        # Play the episode
        for t in range(num_steps_max):
            """
            Epsilon-greedy policy action selection. Then after selection we perform annealing on the epsilon
            factor to encourage exploitation moreso as our model gets trained.
            """
            prob = random.random()
            exp_factor = np.exp(-epsilon_decay_rate * decay_counter)
            epsilon = max_epsilon * exp_factor + min_epsilon * (1 - exp_factor)
            if epsilon > prob:
                action = random.choice(actions)
            else:
                # Compute Q-values via model and take action of highest value
                Q_values = sess.run(
                    model.outputs,
                    feed_dict={model.inputs:state.reshape((1,*state.shape))}
                )
                action_index = np.argmax(Q_values)
                action = actions[action_index]
            """
            Action is chosen. Run the rest of the network!
            """
            decay_counter += 1
            next_frame, reward, done, _ = env.step(action)
            env.render()
            rewards.append(reward)

            if done:
                next_frame = np.zeros((110,84), np.int)
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
                episode_reward = np.sum(rewards)
                print("episode %d: reward = %.4f / loss = %.4f" % (episode, episode_reward, loss))
            else:
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
            memory.add_to_memory((state, action, reward, next_state, done))
            state = next_state

            # Train the network following the Nature article
            train_batch = memory.sample(minibatch_size)

            # Separate the minibatch of experiences into their component parts
            train_states = np.array([x[0] for x in train_batch], ndmin=3)
            train_actions = np.array([x[1] for x in train_batch])
            train_rewards = np.array([x[2] for x in train_batch])
            train_next_states = np.array([x[3] for x in train_batch], ndmin=3)
            train_dones = np.array([x[4] for x in train_batch])

            Q_targets = []
            next_state_Q = sess.run(
                model.outputs,
                feed_dict={model.inputs: train_next_states}
            )
            for i in range(len(train_batch)):
                if train_dones[i]:
                    # If the episode is done, we just assign for the Q-value its reward
                    Q_targets.append(train_rewards[i])
                else:
                    bellman_Q = train_rewards[i] + discount_factor * np.max(next_state_Q[i])
                    Q_targets.append(bellman_Q)
            # Feed targets into DQN and spit out loss
            loss, _ = sess.run(
                [model.loss, model.optimizer],
                feed_dict={
                    model.inputs: train_states,
                    model.actions: train_actions,
                    model.target_Q: np.array(Q_targets)
                }
            )

            # Don't forget-- if our done flag is tripped, break.
            if done:
                break

        # Save point for model. It would really suck if we had to redo this every single time.
        if episode % 5 == 0:
            path = checkpoint.save(sess, "./models/model.ckpt")
            print("saved your model for ya, chief.")

"""
After training, we'll run the game. This just follows the doom-test.py file, except this time actions are
determined by the policy governed by the DQN.
"""
run_flag = False
num_test_episodes = 20

if run_flag:
    checkpoint.restore(sess, "./models/model.ckpt")
    # Start the game
    for i in range(num_test_episodes):
        frame = env.reset()
        state, frames_queue = create_state(frames_queue, frame, True) 
        done = False
        total_reward = 0
        while not done:
            Q_vals = sess.run(
                model.outputs,
                feed_dict={model.inputs: state.reshape((1, *state.shape))}
            )
            action_index = np.argmax(Q_vals)
            action = actions[action_index]
            next_frame, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward

            if done:
                break
            else:
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
                state = next_state
        
        print("total reward is: ", total_reward)
    env.close()