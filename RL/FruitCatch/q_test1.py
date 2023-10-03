import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import pickle
import os

from fruit_catch import FruitCatch
from q_agent import Q_Agent
from q_agent import encode_state
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

max_timestamp = 10_000#_000
eps_drop_freq = 1_500_000
peek_every = 5_000_000
peek_len = 200
print_every = 10_000

# create agent instance with random Q table
agent_file = 'agent_q_test1.pkl'
agent_dir = '.'
load_saved = True
save_after_train = False

if load_saved and agent_file:
    with open(os.path.join(agent_dir, agent_file),'rb') as file:
        myAgent = pickle.load(file)
        print(f"Loaded agent from {agent_file}")
else:
    myAgent = Q_Agent(grid_size, glove_width)
    myAgent.eps_decay = 0.5

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
    state = game.get_state()
    
    # agent decides the next action
    action = myAgent.take_action(state)
    
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

    # print to output
    if game.timestamp % print_every == 0:
        print(f"T={game.timestamp} :: avg success rate={success_rate[-1]}")

    # epsilon decay in epsilon-greedy policy
    if game.timestamp % eps_drop_freq == 0 and game.timestamp > 0:
        myAgent.drop_eps()

    # terminate after max_timestamp reached
    if game.timestamp > max_timestamp:
        game.terminate()

# Save agent after training
if save_after_train:
    with open(os.path.join(agent_dir, agent_file), 'wb') as file:
        pickle.dump(myAgent, file)
        print(f"Saved agent to {agent_file}")

# Plot the success rate over timestamp
if myAgent.training == True:
    fig = plt.figure()
    #plt.set(title='Success rate with time')
    #plt.set(xlabel='Timestamp')
    #plt.set(ylabel='Success rate (%)')
    plt.plot(stat_timestamp, np.array(success_rate)*100.0)
    plt.show()

# Plot the Q table data
if False:
    plot_cols = []
    if game.force_col:
        plot_cols.append(game.force_col)
    else:
        plot_cols = [ math.floor(x) for x in list(np.linspace(0.,0.99*grid_size,3)) ]

    for col in plot_cols:
        # Get Q data with (row=glove_pos, col=fruit_height, depth=action)
        Q_data = np.zeros((game.glove_pos_range[1]+1, grid_size, 3))
        for glove_pos in range(game.glove_pos_range[1]+1):
            for i in range(grid_size):
                Q_data[glove_pos, i, :] = myAgent.Q_table.get(encode_state(glove_pos, (i,col)))

        fig,axs = plt.subplots(1,3)
        fig.suptitle(f'Col={col}')
        ims = []
        vmin = Q_data.min()
        vmax = Q_data.max()
        for i in range(3):
            ims.append(axs[i].imshow(Q_data[:,:,i].transpose((1,0)), vmin=vmin, vmax=vmax))
            axs[i].set(title=f'Action={myAgent.action_space[i]}')
            axs[i].set(xlabel='Glove pos')
            axs[i].set(ylabel='Fruit height')
        cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.4])
        fig.colorbar(ims[0], cax=cbar_ax)
        plt.show()

