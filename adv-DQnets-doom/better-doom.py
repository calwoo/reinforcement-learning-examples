import numpy as np 
import random
import tensorflow as tf
from vizdoom import *
import skimage
import time
from collections import deque
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

"""
We're gonna upgrade the DQN with some extra upgrades so it performs better on environments with more temporally-sparse rewards.
Some improvements of this sort include fixed Q-targets, double/dueling DQN architectures, and enhancements to the experience
replay mechanic. First we import all the boilerplate related to running the Doom game.
"""
# First we gotta get Doom working.
# The vizdoom library allows us to get a game working with configuration and scenario files.
def make_environment():
    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()
    # Possible actions our MDP can take
    left = [1, 0, 0, 0, 0, 0, 0]
    right = [0, 1, 0, 0, 0, 0, 0]
    shoot = [0, 0, 1, 0, 0, 0, 0]
    forward = [0, 0, 0, 1, 0, 0, 0]
    backward = [0, 0, 0, 0, 1, 0, 0]
    turn_left = [0, 0, 0, 0, 0, 1, 0]
    turn_right = [0, 0, 0, 0, 0, 0, 1]
    actions = [left, right, shoot, forward, backward, turn_left, turn_right]
    return game, actions

game, actions = make_environment()

# STORE HYPERPARAMETERS HERE
state_dims = [100,120,4]
frame_stack_size = 4 # as per the Nature article
num_actions = game.get_available_buttons_size()
num_episodes = 200
num_steps_max = 100
minibatch_size = 32
learning_rate = 0.0005
discount_factor = 0.95
max_epsilon = 1.0 # epsilon greedy parameters
min_epsilon = 0.01
epsilon_decay_rate = 0.0001
memory_size = 1000000 # memory size from Nature article for experience replay

"""
To use our deep Q-network, we will preprocess the game frames because they have many undesirable properties as
they are-- large resolution, color data (the agent doesn't need this). This makes computation faster.
"""
def preprocess_frame(frame):
    cropped_frame = frame[15:-5, 20:-20]
    normalized_frame = cropped_frame / 255.0
    pp_frame = skimage.transform.resize(normalized_frame, [100,120])
    return pp_frame

"""
Our states will follow the outline in the 2015 Nature article on DQN by Google Deepmind. There they used stacked
frames as states (ie, bundles of consecutive frames). This is necessary, as without this the network could not
learn the motion of enemies and (later) projectiles.
"""
frames_queue = deque([np.zeros((100,120), dtype=np.int) for _ in range(frame_stack_size)], 
    maxlen=frame_stack_size)
def create_state(frames_queue, current_frame, new_epi_flag=False):
    frame = preprocess_frame(current_frame)
    # If new episode, restart the frames_queue with the current frame
    if new_epi_flag:
        frames_queue = deque([np.zeros((100,120), dtype=np.int) for _ in range(frame_stack_size)], 
            maxlen=frame_stack_size)
        for _ in range(frame_stack_size):
            frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    else:
        frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    return state, frames_queue

"""
Next up is to construct the DQN and memory architectures. We will still use a DQN architecture as in the Deepmind
Nature article, but with some twists to speed up computation. Indeed, the biggest twist is an implementation of
the dueling DQN architecture.

The main idea behind this is that the value-action function Q(s,a) can be decomposed into two terms:
    Q(s,a) = A(s,a) + V(s)
where A(s,a) is the "advantage" of taking the action a in state s (over all other actions). Hence we can let our
DQN network become an estimator of both the value function V(s) and the advantage A(s,a). We then combine these
estimators to get our estimates of Q(s,a).

Why do this? Because Q(s,a) by itself doesn't tell me how good an action possibly is. If the value of a state is
very high to begin with, then a high Q(s,a) doesn't mean much. We usually want to choose actions that give us
the best increase to our value, ie, those with the highest advantage. By separately learning these advantages,
the network can start to choose actions with more bang-for-the-buck as opposed to just bang.
"""
class DDQN:
    def __init__(self, num_actions, state_dims, learning_rate):
        self.num_actions = num_actions
        self.state_dims = state_dims
        self.learning_rate = learning_rate

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
        self.conv1_relu = tf.nn.relu(self.conv1)
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1_relu,
            filters=64,
            kernel_size=[4,4],
            strides=[2,2],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        self.conv2_relu = tf.nn.relu(self.conv2)
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2_relu,
            filters=128,
            kernel_size=[3,3],
            strides=[2,2],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        self.conv3_relu = tf.nn.relu(self.conv3)
        self.flatten = tf.layers.flatten(self.conv3_relu)

        # Value branch
        self.value_fc_layer = tf.layers.dense(
            inputs=self.flatten,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value_output = tf.layers.dense(
            inputs=self.value_fc_layer,
            units=1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Advantage branch
        self.advantage_fc_layer = tf.layers.dense(
            inputs=self.flatten,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.advantage_output = tf.layers.dense(
            inputs=self.advantage_fc_layer,
            units=num_actions,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Combine the two to get Q-values via the rule
        #   Q(s,a) = V(s) + (A(s,a) - 1/num_actions * sum of A(s,a'))
        self.Q_values = self.value_output + tf.subtract(
            self.advantage_output,
            tf.reduce_mean(self.advantage_output, axis=1, keepdims=True)
        )

        # Loss and absolute error for use in PER.
        self.error = tf.abs(self.Q_values - self.target_Q)
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
        # Optimizer
        self.ISweight = tf.placeholder(tf.float32, shape=[None, 1])
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

"""
The second main improvement is that we do prioritized experience replay. The issue with vanilla experience replay
is that more important experiences are given the same probabilistic chance of occuring as less important ones,
meaning that it takes many more samples to learn and the variance of samples is high.

PER fixes this by changing the sampling distribution. The slogan is that we want to sample experience with a large
difference between our prediction and the TD target, as those are the experiences that give us a lot to learn.
"""
class SumTree:
    # Data structure like a binary tree where nodes are sum of leaves. Implemented as array.
    data_index = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity)

    def add(self, priority, data):
        index = self.capacity - 1 + data_index
        self.data[data_index] = data
        self.update(index, priority)
        data_index += 1
        if data_index == capacity:
            data_index = 0

    def update(self, index, priority):
        pass





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
    
    def initialize_memory(self, game, actions, frames_queue):
        # Clear memory and start over
        self.memory.clear()
        game.new_episode()
        # Create initial batch of experiences via random actions
        for i in range(minibatch_size):
            if i == 0:
                frame = game.get_state().screen_buffer
                state, frames_queue = create_state(frames_queue, frame, True)
            action = random.choice(actions)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                next_state = np.zeros(state.shape)
                self.add_to_memory((state, action, reward, next_state, done))
                game.new_episode()
                frame = game.get_state().screen_buffer
                state, frames_queue = create_state(frames_queue, frame, True)
            else:
                next_frame = game.get_state().screen_buffer
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
                self.add_to_memory((state, action, reward, next_state, done))
                state = next_state
        return frames_queue