import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import pickle
import os

from fruit_catch import FruitCatch
from dqn_agent import *
from scalar_cache import *

# game settings
grid_size = 20
glove_width = 3
prob_extra = 0.

# create game instance
game = FruitCatch(grid_size, glove_width, prob_extra)

# configure some game settings
game.fps = 50
#game.force_col = int(grid_size/2)
game.fruit_miss = -10.
game.fruit_reward = 10.
game.move_reward = -0.1

iter_model_every = 10_000
train_model_every = 1_000

max_timestamp = 10_000_000
eps_drop_freq = 1_500_000
peek_every = 5_000_000
peek_len = 200
print_every = 10_000

myAgent = DQN_Agent(grid_size, glove_width)

# run training session
myAgent.training = True

display_override = False
display_flag = True

# stats
stat_timestamp = []
stat_log_every = 10
success_cache = ScalarCache(100)
success_rate = []

game.start_environment()

max_timestamp += peek_len
while game.running == True:
    # get state from environment
    observation = game.get_grid()
    
    # agent decides the next action
    action = myAgent.take_action(observation)
    
    # decide whether display should be updated
    display_flag = (game.timestamp % peek_every < peek_len)

    # step the game (update display given display_flag)
    reward, success = game.step(action, display_flag)

    myAgent.receive_reward(reward)

    # calculate stats 
    if success==1:
        success_cache.append(1)
    elif success==-1:
        success_cache.append(0)
    if game.timestamp % stat_log_every == 0:
        success_rate.append(success_cache.avg())
        stat_timestamp.append(game.timestamp)

    # train model parameters
    if game.timestamp % train_model_every == 0:
        print("Trained predictor using batch")
        myAgent.train_predictor()

    # iterate model parameters
    if game.timestamp % iter_model_every == 0:
        print("Iterated model (predictor model params --> target model params)")
        myAgent.iterate_model()

    # print to output
    if game.timestamp % print_every == 0:
        print(f"T={game.timestamp} :: avg success rate={success_rate[-1]}")

    # epsilon decay in epsilon-greedy policy
    if game.timestamp % eps_drop_freq == 0 and game.timestamp > 0:
        myAgent.drop_eps()

    # terminate after max_timestamp reached
    if game.timestamp > max_timestamp:
        game.terminate()


# Plot the success rate over timestamp
#if myAgent.training == True:
fig = plt.figure()
#plt.set(title='Success rate with time')
#plt.set(xlabel='Timestamp')
#plt.set(ylabel='Success rate (%)')
plt.plot(stat_timestamp, np.array(success_rate)*100.0)
plt.show()