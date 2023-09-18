import numpy as np
import random
import math
import pygame
from pygame.locals import *
from cache import ScalarCache

class Catch:
    def __init__(self, grid_size:int, glove_width:int, prob_extra:float):
        self.fps = 60                       # frame rate of simulation
        self.move_reward = -0.01              # reward for moving (action -1 or 1)
        self.fruit_reward = 1.0             # reward for catching fruit
        self.fruit_miss = -1.0              # reward for missing fruit
        self.grid_size = grid_size
        self.glove_width = glove_width                  
        self.prob_extra = prob_extra        # probability of extra fruit spawning (0,...,0.2)
        self.max_timestamp = 100000           # simulation stops after max_timestamp if not None
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
    def run(self, get_action=None, send_reward=None, show_display:bool=True, manual:bool=True, agent_input:str=''):
        self.grid_unit = 20
        self.window_width, self.window_height = self.grid_unit*self.grid_size, self.grid_unit*self.grid_size
        
        if show_display:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Catch")
            clock = pygame.time.Clock()

        action = 0
        display_on = False
                
        self.running = True
        while self.running:
            # update environment observation based on current state
            if display_on:
                pygame.display.update()
                clock.tick(self.fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                keys_pressed = pygame.key.get_pressed()

                self.update_display()

            # get key inputs to get action if in manual mode
            if display_on and manual:
                if keys_pressed[pygame.K_a] and not keys_pressed[pygame.K_d]:
                    action = -1
                elif keys_pressed[pygame.K_d] and not keys_pressed[pygame.K_a]:
                    action = 1
                else:
                    action = 0
            elif not manual:
            # if not manual then send observation to agent
            # to get the agent's action
                if agent_input == 'Screenshot' and display_on:
                    # sends a screenshot img to agent
                    # requires agent already knowing width,height of img
                    pass # not yet implemented
                elif agent_input == 'Grid':
                    # sends pixel grid to agent
                    # requires agent already knowing grid size
                    action = get_action(self.grid)
                elif agent_input == 'Data':
                    # note: this config only works if prob_extra=0 (i.e. only one fruit at any time)
                    # sends glove position and ONLY fruit position to agent
                    # requires agent already knowing grid size, glove size/range
                    action = get_action((self.glove_pos, (self.fruit[0][0], self.fruit[0][1]) if len(self.fruit) > 0 else None))
            
            # get next environment state
            reward = self.update_state(action)

            # send reward to agent
            if not manual:
                send_reward(reward)

            # update recorded stats
            self.update_stats()

            # print progress
            if self.timestamp % 1000 == 0:
                print(f'timestamp={self.timestamp}, success rate={round(self.success_cache.avg(),3)}')

            # display periodically
            if show_display and 0 <= (self.timestamp % 100000) <= 1000:
                display_on = True
            else:
                display_on = False

            self.timestamp += 1
            if self.max_timestamp and self.timestamp > self.max_timestamp:
                self.running = False