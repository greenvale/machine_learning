import numpy as np
import random
import math
import pygame
from pygame.locals import *
from scalar_cache import ScalarCache

class FruitCatch:
    def __init__(self, grid_size:int, glove_width:int, prob_extra:float):
        self.fps = 60                       # frame rate of simulation
        self.move_reward = -0.01              # reward for moving (action -1 or 1)
        self.fruit_reward = 1.0             # reward for catching fruit
        self.fruit_miss = -1.0              # reward for missing fruit
        self.grid_size = grid_size
        self.glove_width = glove_width                  
        self.prob_extra = prob_extra        # probability of extra fruit spawning (0,...,0.2)
        self.force_col = None               # forces fruit to fall from one col if not None

        # setup the game attributes
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.glove_pos = 0
        self.glove_pos_range = (0, grid_size - glove_width)
        self.fruit = []
        self.timestamp = 0
        self.running = False

        # stats
        self.stat_timestamp = []
        self.stat_log_inc = 10
        self.success_cache = ScalarCache(100)
        self.success_rate = []

        # training settings
        self.max_timestamp = None
        self.eps_drop_freq = None
        self.peek_every = None
        self.peek_len = None
        self.display_override = False
        self.final_peek = True

    # take action input and determine the new state and reward    
    def update_state(self, action:int) -> float:
        reward = 0.

        # move fruit
        delete_cache = []
        for i in range(len(self.fruit)):
            self.fruit[i][0] += 1
            if self.fruit[i][0] == self.grid_size:
                if self.glove_pos <= self.fruit[i][1] < self.glove_pos + self.glove_width:
                    reward += self.fruit_reward
                    self.success_cache.append(1)
                else:
                    reward += self.fruit_miss
                    self.success_cache.append(0)
                delete_cache.append(i)
        
        # delete fruit that have been marked in cache
        for i in delete_cache:
            del self.fruit[i]
        delete_cache.clear()

        # move glove
        if action == -1 and self.glove_pos > self.glove_pos_range[0]:
            self.glove_pos -= 1
            reward += self.move_reward
        elif action == 1 and self.glove_pos < self.glove_pos_range[1]:
            self.glove_pos += 1
            reward += self.move_reward
        
        # roll dice to decide if new piece of fruit is to be spawned
        roll = False
        if self.prob_extra > 0:
            inv = int(math.ceil(1.0 / self.prob_extra))
            roll = (random.randint(0, inv) == 0)

        # spawn fruit if necessary
        if len(self.fruit)==0 or roll:
            col = random.randint(0, self.grid_size-1)
            if self.force_col != None:
                col = self.force_col
            self.fruit.append([0, col])

        return reward
    
    # updates the grid representation of environment
    def update_grid(self):
        # set grid to all zeros to refresh
        self.grid[:,:] = 0

        # draw glove
        self.grid[-1,self.glove_pos:self.glove_pos+self.glove_width] = 1

        # draw fruit
        for f in self.fruit:
            self.grid[f[0], f[1]] = 1

    # updates the logged stats
    def update_stats(self):
        if self.timestamp % self.stat_log_inc:
            self.stat_timestamp.append(self.timestamp)
            self.success_rate.append(self.success_cache.avg())
            
    # updates the UI display representation of environment
    def update_display(self):
        self.window.fill((0,0,0))

        # draw glove
        glove_coords = np.array([self.glove_pos, self.grid_size-1, self.glove_width, 1])*self.grid_unit
        pygame.draw.rect(self.window, (255,255,0), pygame.Rect( *(list(glove_coords)) ))

        # draw fruit
        for f in self.fruit:
            fruit_coords = np.array([f[1], f[0], 1, 1])*self.grid_unit
            pygame.draw.rect(self.window, (255,0,0), pygame.Rect( *(list(fruit_coords)) ))

        # text
        sysfont = pygame.font.get_default_font()
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'rate={round(self.success_cache.avg(),3)}', True, (255,255,255))
        self.window.blit(img, (20, 20))

    # launches the game environment loop
    # manages client-game interaction for manual control or agent input
    def run(self, agent=None, agent_input:str=''):
        self.grid_unit = 20
        self.window_width, self.window_height = self.grid_unit*self.grid_size, self.grid_unit*self.grid_size

        # display on flag
        # switched on/off if peeking or not
        # initially set to on
        display_flag = True
        
        # initialise pygame window
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Catch")
        clock = pygame.time.Clock()

        action = 0
        init_display_override = None
                
        self.running = True
        while self.running:
            # update environment observation based on current state
            if display_flag:
                pygame.display.update()
                clock.tick(self.fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                keys_pressed = pygame.key.get_pressed()

                self.update_display()

            if agent:
            # if not manual then send observation to agent
            # to get the agent's action
                if agent_input == 'Screenshot' and display_flag:
                    # sends a screenshot img to agent
                    # requires agent already knowing width,height of img
                    pass # not yet implemented
                elif agent_input == 'Grid':
                    # sends pixel grid to agent
                    # requires agent already knowing grid size
                    action = agent.take_action(self.grid)
                elif agent_input == 'Data':
                    # note: this config only works if prob_extra=0 (i.e. only one fruit at any time)
                    # sends glove position and ONLY fruit position to agent
                    # requires agent already knowing grid size, glove size/range
                    action = agent.take_action((self.glove_pos, (self.fruit[0][0], self.fruit[0][1]) if len(self.fruit) > 0 else None))
            else:
            # get key inputs to get action if in manual mode
                if keys_pressed[pygame.K_a] and not keys_pressed[pygame.K_d]:
                    action = -1
                elif keys_pressed[pygame.K_d] and not keys_pressed[pygame.K_a]:
                    action = 1
                else:
                    action = 0

            # get next environment state
            reward = self.update_state(action)

            # send reward to agent
            if agent:
                agent.receive_reward(reward)

            # update recorded stats
            self.update_stats()

            # print progress
            if self.timestamp % 10000 == 0:
                print(f'timestamp={self.timestamp}, success rate={round(self.success_cache.avg(),3)}, agent eps={agent.eps if agent.eps else "N/a"}')

            # epsilon decay
            if agent.training==True and self.timestamp > 100 and (self.timestamp % self.eps_drop_freq == 0):
                agent.drop_eps()

            # display periodically
            if  0 <= (self.timestamp % self.peek_every) < self.peek_len:
                display_flag = True
            else:
                display_flag = False or (self.display_override==True)

            # advance timestamp
            self.timestamp += 1
            if self.max_timestamp and (self.timestamp > self.max_timestamp):
                if self.final_peek==True:
                    init_display_override = self.display_override # store initial display override setting
                    self.max_timestamp += self.peek_len
                    self.display_override = True
                    self.final_peek = False
                else:
                    self.running = False
                    self.display_override = init_display_override