from vizdoom import *
import random
import time

# Standard game initializer
game = DoomGame()
game.load_config("basic.cfg")
game.set_doom_scenario_path("basic.wad")
game.init()

# Set up actions for agent to use
left = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]
actions = [left, right, shoot]

# Run through a random choice policy for some episodes
num_episodes = 10
for episode in range(num_episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        vars = state.game_variables
        action = random.choice(actions)
        reward = game.make_action(action)
        print("state %d" % state.number)
        print("vars = ", vars)
        print("reward = ", reward)
        time.sleep(0.02)
    print("total_rewards = ", game.get_total_reward())
    time.sleep(2)