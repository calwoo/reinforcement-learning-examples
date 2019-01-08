import numpy as np 
import random
import tensorflow as tf
from vizdoom import *
import skimage
import time
from collections import deque
import warnings
from progress.bar import Bar

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
num_episodes = 3000
num_steps_max = 4000
minibatch_size = 64
learning_rate = 0.0005
discount_factor = 0.95
max_epsilon = 1.0 # epsilon greedy parameters
min_epsilon = 0.01
epsilon_decay_rate = 0.0001
memory_size = 20000 # memory size from Nature article for experience replay
transfer_max = 10000

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
    def __init__(self, num_actions, state_dims, learning_rate, name):
        self.num_actions = num_actions
        self.state_dims = state_dims
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(self.name):

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
            self.output = self.value_output + tf.subtract(
                self.advantage_output,
                tf.reduce_mean(self.advantage_output, axis=1, keepdims=True)
            )
            self.Q_values = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)

            # Loss and absolute error for use in PER.
            self.ISweight = tf.placeholder(tf.float32, shape=[None, 1])
            self.error = tf.abs(self.Q_values - self.target_Q)
            self.loss = tf.reduce_mean(self.ISweight * tf.square(self.target_Q - self.Q_values))
            # Optimizer
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
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        index = self.capacity - 1 + self.data_index
        self.data[self.data_index] = data
        self.update(index, priority)
        self.data_index += 1
        if self.data_index == self.capacity:
            self.data_index = 0

    def update(self, index, priority):
        delta = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get_leaf(self, v):
        parent = 0
        while True:
            # If at leaf, terminate
            if parent >= self.capacity - 1:
                leaf_ind = parent
                break
            left_child = parent * 2 + 1
            right_child = left_child + 1
            # Descend.
            if self.tree[left_child] < v:
                parent = right_child
                v -= self.tree[left_child]
            else:
                parent = left_child
        data_ind = leaf_ind - (self.capacity - 1)
        return leaf_ind, self.tree[leaf_ind], self.data[data_ind]

    def total_priority(self):
        return self.tree[0]

class Memory:
    # PER hyperparameters
    PER_alpha = 0.5
    PER_beta = 0.3
    PER_beta_annealing = 0.001
    PER_epsilon = 0.01
    lowest_error = 1

    def __init__(self, memory_size):
        self.memory = SumTree(memory_size)
        self.memory_size = memory_size

    def add_to_memory(self, experience):
        # Add to sumtree with maximal priority (just initially, the priority will be overwritten
        # when we sample it again using the TD error)
        max_priority = np.max(self.memory.tree[self.memory_size:])
        if max_priority == 0:
            max_priority = self.lowest_error
        self.memory.add(max_priority, experience)

    def sample(self, minibatch_size):
        # As per the PER paper, to sample a minibatch of size k, we split up [0, total_priority] into k
        # equal parts, and then we sample uniformly from each range. As we are also now sampling from a
        # distribution different from the data distribution, we must correct for this by importance
        # sampling weights.
        minibatch = []
        batch_ids = np.empty((minibatch_size,), dtype=np.int32)
        batch_ISweights = np.empty((minibatch_size, 1), dtype=np.float32)
        priority_seg = self.memory.total_priority() / minibatch_size
        # Perform linear annealing for beta
        self.PER_beta = np.minimum(1.0, self.PER_beta + self.PER_beta_annealing)
        # Max weight for IS update
        min_prob = np.min(self.memory.tree[self.memory_size:]) / self.memory.total_priority()
        max_weight = np.power((min_prob * self.memory_size), -self.PER_beta)
        # Sample the minibatch
        for i in range(minibatch_size):
            top, bottom = priority_seg * i, priority_seg * (i+1)
            v = np.random.uniform(top, bottom)
            leaf_ind, priority, exp = self.memory.get_leaf(v) 
            IS_prob = priority / self.memory.total_priority()
            batch_ISweights[i,0] = np.power((IS_prob * minibatch_size), -self.PER_beta) / max_weight
            batch_ids[i] = leaf_ind
            minibatch.append(exp)
        return minibatch, batch_ids, batch_ISweights

    def update_memory(self, ids, deltas):
        deltas += self.PER_epsilon
        deltas = np.minimum(deltas, self.lowest_error)
        new_priorities = np.power(deltas, self.PER_alpha)
        for i, p in zip(ids, new_priorities):
            self.memory.update(i, p)

    def initialize_memory(self, game, actions, frames_queue):
        game.new_episode()
        # Progress bar
        bar = Bar("Initializing memory...", max=self.memory_size)
        # Create initial batch of experience via random actions
        for i in range(self.memory_size):
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
            bar.next()
        bar.finish()
        return frames_queue

"""
Now we train our network. Note that we have two DQN networks as we are using the Double DQN architecture as well.
The double DQN aims to solve the "moving target" problem, which is that the Q-update overestimates the values.

The solution is to use two DQNs-- one to select a best action to take for the next state, and another target net
to calculate the target Q-value of taking that action at the next state.
"""
model_net = DDQN(num_actions, state_dims, learning_rate, name="model")
target_net = DDQN(num_actions, state_dims, learning_rate, name="target_net")
memory = Memory(memory_size)
frames_queue = memory.initialize_memory(game, actions, frames_queue)
sess = tf.Session()
checkpoint = tf.train.Saver() # to save model during training

sess.run(tf.global_variables_initializer())
game.init()

## HELPER FUNCTION for transferring weights
def update_tf_graph():
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_net")
    op_holder = []
    # Update target_net with parameters from model.
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

update_target = update_tf_graph()
sess.run(update_target)

# Flag to turn on training or not. If we're just running to evaluate, we should turn this off as we
# will have model parameters saved for the DQN.
training_flag = True
transfer_counter = 0

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
                    model_net.output,
                    feed_dict={model_net.inputs: state.reshape((1,*state.shape))}
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
                next_frame = np.zeros((100,120), np.int)
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
            else:
                next_frame = game.get_state().screen_buffer
                next_state, frames_queue = create_state(frames_queue, next_frame, False)
            memory.add_to_memory((state, action, reward, next_state, done))
            state = next_state

            # Train the network following PER, DDQN specifications
            train_batch, train_batch_ids, train_batch_ISweights = memory.sample(minibatch_size)

            # Separate the minibatch of experiences into their component parts
            train_states = np.array([x[0] for x in train_batch], ndmin=3)
            train_actions = np.array([x[1] for x in train_batch])
            train_rewards = np.array([x[2] for x in train_batch])
            train_next_states = np.array([x[3] for x in train_batch], ndmin=3)
            train_dones = np.array([x[4] for x in train_batch])

            Q_targets = []
            """
            How double DQN works is that we get our target Q-values via a two-step process:
                1) Use our model network to compute a* = argmax_a Q(s',a)
                2) Use our target to compute Q(s',a*)
                3) After some steps, transfer parameters for the model to the target network.
            """
            next_state_Q = sess.run(
                model_net.output,
                feed_dict={model_net.inputs: train_next_states}
            )
            next_state_Q_targets = sess.run(
                target_net.output,
                feed_dict={target_net.inputs: train_next_states}
            )
            transfer_counter += 1
            for i in range(len(train_batch)):
                if train_dones[i]:
                    # If the episode is done, we just assign for the Q-value its reward
                    Q_targets.append(train_rewards[i])
                else:
                    action = np.argmax(next_state_Q[i])
                    target = train_rewards[i] + discount_factor * next_state_Q_targets[i][action]
                    Q_targets.append(target)
            # Feed targets into DQN and spit out loss
            loss, _, deltas = sess.run(
                [model_net.loss, model_net.optimizer, model_net.error],
                feed_dict={
                    model_net.inputs: train_states,
                    model_net.actions: train_actions,
                    model_net.target_Q: np.array(Q_targets),
                    model_net.ISweight: train_batch_ISweights
                }
            )

            # Update memory priorities
            memory.update_memory(train_batch_ids, deltas)

            # Transfer parameters
            if transfer_counter >= transfer_max:
                transfer_counter = 0
                update_target = update_tf_graph()
                sess.run(update_target)
                print("parameters transferred")

            # Don't forget-- if our done flag is tripped, break.
            if done:
                break
        
        # Print episode info at end.
        episode_reward = np.sum(rewards)
        print("episode %d: reward = %.4f / loss = %.4f" % (episode, episode_reward, loss))

        # Save point for model. It would really suck if we had to redo this every single time.
        if episode % 5 == 0:
            path = checkpoint.save(sess, "./models/model.ckpt")
            print("saved your model for ya, chief.")

"""
After training, we'll run the game.
"""
run_flag = False
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
                model_net.outputs,
                feed_dict={model_net.inputs: state.reshape((1, *state.shape))}
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

