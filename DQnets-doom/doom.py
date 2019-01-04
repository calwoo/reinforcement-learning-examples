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
import skimage
import time
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
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [left, right, shoot]
    return game, actions

game, actions = make_environment()

# STORE HYPERPARAMETERS HERE
state_dims = [84,84,4]
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
    crop = frame[30:-10, 30:-10]
    normalize = crop / 255.0
    pp_frame = skimage.transform.resize(normalize, [84,84])
    return pp_frame

"""
Our states will follow the outline in the 2015 Nature article on DQN by Google Deepmind. There they used stacked
frames as states (ie, bundles of consecutive frames). This is necessary, as without this the network could not
learn the motion of enemies and (later) projectiles.
"""
frames_queue = deque([np.zeros((84,84), dtype=np.int) for _ in range(frame_stack_size)], 
    maxlen=frame_stack_size)
def create_state(frames_queue, current_frame, new_epi_flag=False):
    frame = preprocess_frame(current_frame)
    # If new episode, restart the frames_queue with the current frame
    if new_epi_flag:
        frames_queue = deque([np.zeros((84,84), dtype=np.int) for _ in range(frame_stack_size)], 
            maxlen=frame_stack_size)
        for _ in range(frame_stack_size):
            frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    else:
        frames_queue.append(frame)
        state = np.stack(frames_queue, axis=2)
    return state, frames_queue

"""
A lingering question remains: why didn't people use deep nonlinear neural networks before Deepmind for
Q-learning? After all, function approximation methods have been common in RL for a long time. Quoting the
Nature article:

"Reinforcement learning is known to be unstable or even to diverge when a nonlinear function approximator... is
used to represent the action-value function. This instability has several causes: the correlations present in
the sequence of observations, the fact that small updates to Q may significantly change the policy and therefore
change the data distribution, and the correlations between the action-values and the target values.

We address these instabilities with a novel varient of Q-learning, which uses two key ideas. First, we used a
biologically inspired mechanism termed experience replay that randomizes over data, thereby removing correlations
in the observation sequence and smoothing over changes in the data distribution. Second, we used an iterative
update that adjusts the action-values towards target values that ae only periodically updated, thereby reducing
correlations with the target."
"""
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

"""
Experience replay is the following idea: Given the large state space of some environments, it is highly unlikely
to revisit a scenario multiple times in succession. However, to keep learning from this situation and to prevent
divergent behavior of the function approximator because of correlated states, we keep a memory bank of 
"past experiences" and on each training loop, perform the Q-learning update rule on each of these past exps.
"""
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

"""
Now we train the network.
"""
model = NatureDQN(state_dims, num_actions, learning_rate)
memory = Memory(memory_size)
frames_queue = memory.initialize_memory(game, actions, frames_queue)
sess = tf.Session()
checkpoint = tf.train.Saver() # to save model during training

sess.run(tf.global_variables_initializer())
game.init()

# Flag to turn on training or not. If we're just running to evaluate, we should turn this off as we
# will have model parameters saved for the DQN.
training_flag = False

if training_flag:
    for episode in range(num_episodes):
        rewards = []
        decay_counter = 0
        game.new_episode()
        frame = game.get_state().screen_buffer
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
            reward = game.make_action(action)
            done = game.is_episode_finished()
            rewards.append(reward)

            if done:
                next_frame = np.zeros((84,84), np.int)
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
                episode_reward = np.sum(rewards)
                print("episode %d: reward = %.4f / loss = %.4f" % (episode, episode_reward, loss))
            else:
                next_frame = game.get_state().screen_buffer
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
run_flag = True
num_test_episodes = 20

if run_flag:
    game.init()
    checkpoint.restore(sess, "./models/model.ckpt")
    # Start the game
    for i in range(num_test_episodes):
        game.new_episode()
        frame = game.get_state().screen_buffer
        state, frames_queue = create_state(frames_queue, frame, True) 
        while not game.is_episode_finished():
            Q_vals = sess.run(
                model.outputs,
                feed_dict={model.inputs: state.reshape((1, *state.shape))}
            )
            action_index = np.argmax(Q_vals)
            action = actions[action_index]
            game.make_action(action)

            done = game.is_episode_finished()

            if done:
                break
            else:
                next_frame = game.get_state().screen_buffer
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
                state = next_state
        score = game.get_total_reward()
        print("final score is: ", score)
    game.close()