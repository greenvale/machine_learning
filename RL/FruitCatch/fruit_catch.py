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

    # take action input and determine the new state and reward    
    # expects actions in {-1, 0, 1}
    def update_state(self, action:int) -> (float, float):
        reward = 0.
        success = None

        # move fruit
        delete_cache = []
        for i in range(len(self.fruit)):
            self.fruit[i][0] += 1
            if self.fruit[i][0] == self.grid_size:
                if self.glove_pos <= self.fruit[i][1] < self.glove_pos + self.glove_width:
                    reward += self.fruit_reward
                    success = 1
                else:
                    reward += self.fruit_miss
                    success = -1
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

        return reward, success
    
    # updates the grid representation of environment
    def update_grid(self) -> None:
        # set grid to all zeros to refresh
        self.grid[:,:] = 0

        # draw glove
        self.grid[-1,self.glove_pos:self.glove_pos+self.glove_width] = 1

        # draw fruit
        for f in self.fruit:
            self.grid[f[0], f[1]] = 1
            
    # updates the UI display representation of environment
    def update_display(self) -> None:
        self.window.fill((0,0,0))

        # draw glove
        glove_coords = np.array([self.glove_pos, self.grid_size-1, self.glove_width, 1])*self.grid_unit
        pygame.draw.rect(self.window, (255,255,0), pygame.Rect( *(list(glove_coords)) ))

        # draw fruit
        for f in self.fruit:
            fruit_coords = np.array([f[1], f[0], 1, 1])*self.grid_unit
            pygame.draw.rect(self.window, (255,0,0), pygame.Rect( *(list(fruit_coords)) ))

        # text
        #sysfont = pygame.font.get_default_font()
        #font = pygame.font.SysFont(None, 24)
        #img = font.render(f'rate={round(self.success_cache.avg(),3)}', True, (255,255,255))
        #self.window.blit(img, (20, 20))

    # setup the environment
    def start_environment(self) -> None:
        self.grid_unit = 20
        self.window_width, self.window_height = self.grid_unit*self.grid_size, self.grid_unit*self.grid_size

        # initialise pygame window
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Catch")
        self.clock = pygame.time.Clock()
        self.timestamp = 0
        self.running = True

    # step the environment by taking action
    # update the state
    # update the display if required
    def step(self, action:int, display_flag:bool=True) -> (float, float):
        reward, success = self.update_state(action)

        if display_flag:
            pygame.display.update()
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.update_display()
        self.timestamp += 1

        return reward, success

    # returns the grid representation of the environment by reference
    def get_grid(self) -> np.array:
        self.update_grid()
        return self.grid

    # returns the state of the game
    # only should be run if there is maximum 1 fruit at any one time
    # otherwise data is invalid
    def get_state(self):
        assert(len(self.fruit) <= 1)
        return (self.glove_pos, (self.fruit[0][0], self.fruit[0][1]) if len(self.fruit) > 0 else None)
    
    # terminates the environment
    def terminate(self):
        self.running = False