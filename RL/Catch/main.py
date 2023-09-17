from catch import Catch
from agent import Agent

grid_size = 20
glove_width = 3
prob_extra = 0.

game = Catch(grid_size, glove_width, prob_extra)

myAgent = Agent(grid_size, glove_width)

game.run(myAgent.take_action, myAgent.receive_reward, False, 'Data')