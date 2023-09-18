import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from catch import Catch
from agent import QRL_Agent
from agent import encode_state

# game settings
grid_size = 20
glove_width = 3
prob_extra = 0.

# create game instance
game = Catch(grid_size, glove_width, prob_extra)

# configure some game settings
game.fps = 60
game.max_timestamp = 100000
game.force_col = int(grid_size/2)
game.fruit_miss = -10.
game.fruit_reward = 10.
game.move_reward = -0.01

# create agent instance
myAgent = QRL_Agent(grid_size, glove_width)

# run training session
game.run(myAgent.take_action, myAgent.receive_reward, False, False, 'Data')

print("Finished training")

# Plot the success rate over timestamp
fig = plt.figure()
#plt.set(title='Success rate with time')
#plt.set(xlabel='Timestamp')
#plt.set(ylabel='Success rate (%)')
plt.plot(game.stat_timestamp, np.array(game.success_rate)*100.0)
plt.show()

# Plot the Q table data
# Get Q data with (row=glove_pos, col=fruit_height, depth=action)
Q_data = np.zeros((game.glove_pos_range[1]+1, grid_size, 3))
for glove_pos in range(game.glove_pos_range[1]+1):
    for i in range(grid_size):
        Q_data[glove_pos, i, :] = myAgent.Q_table.get(encode_state(glove_pos, (i,game.force_col)))

fig,axs = plt.subplots(1,3)
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

