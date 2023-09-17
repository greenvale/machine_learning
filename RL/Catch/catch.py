import numpy as np
import random
import math
import pygame

class Catch:
    def __init__(self, size:int, width:int, prob_extra:float):
        self.action_space = [-1, 0, 1]
        self.fps = 20
        self.move_cost = -0.01
        self.fruit_reward = 1.0
        self.fruit_miss = -1.0
        self.size = size
        self.width = width
        self.prob_extra = prob_extra
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.pos = 0
        self.pos_range = (0, size-width)
        self.fruit = []
        self.counter = 0
        self.running = False
    
    def update_state(self, action:int) -> float:
        reward = 0.
        # move fruit
        delete_cache = []
        for i in range(len(self.fruit)):
            self.fruit[i][0] += 1
            if self.fruit[i][0] == self.size:
                if self.pos <= self.fruit[i][1] < self.pos + self.width:
                    print("caught fruit")
                    reward += self.fruit_reward
                else:
                    print("didn't catch fruit")
                    reward += self.fruit_miss
                delete_cache.append(i)
        
        # delete fruit that have been marked in cache
        for i in delete_cache:
            del self.fruit[i]
        delete_cache.clear()

        # move glove
        if action == -1 and self.pos > self.pos_range[0]:
            self.pos -= 1
            reward += self.move_cost
        elif action == 1 and self.pos < self.pos_range[1]:
            self.pos += 1
            reward += self.move_cost
        
        # roll dice to decide if new piece of fruit is required
        roll = False
        if self.prob_extra > 0:
            inv = int(math.ceil(1.0 / self.prob_extra))
            roll = (random.randint(0, inv) == 0)

        # create fruit if necessary
        if len(self.fruit)==0 or roll:
            col = random.randint(0, self.size-1)
            self.fruit.append([0, col])
        
        self.counter += 1

        return reward

    def update_display(self):
        ## --- Update grid ---------------------------
        # set grid to all zeros to refresh
        self.grid[:,:] = 0

        # draw glove
        self.grid[-1,self.pos:self.pos+self.width] = 1

        # draw fruit
        for f in self.fruit:
            self.grid[f[0], f[1]] = 1
        
        ## --- Update gui ---------------------------
        self.window.fill((0,0,0))

        # draw glove
        glove_coords = np.array([self.pos, self.size-1, self.width, 1])*self.grid_unit
        pygame.draw.rect(self.window, (255,255,0), pygame.Rect( *(list(glove_coords)) ))

        # draw fruit
        for f in self.fruit:
            fruit_coords = np.array([f[1], f[0], 1, 1])*self.grid_unit
            pygame.draw.rect(self.window, (255,0,0), pygame.Rect( *(list(fruit_coords)) ))

    
    def run(self, get_action=None, send_reward=None, manual:bool=True, agent_input:str=''):
        self.grid_unit = 20
        self.window_width, self.window_height = self.grid_unit*self.size, self.grid_unit*self.size
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Catch")
        
        action = 0
        
        clock = pygame.time.Clock()
        self.running = True
        while self.running:
            clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            keys_pressed = pygame.key.get_pressed()

            # update environment observation based on current state
            self.update_display()

            # get key inputs to get action if in manual mode
            if manual:
                if keys_pressed[pygame.K_a] and not keys_pressed[pygame.K_d]:
                    action = self.action_space[0]
                elif keys_pressed[pygame.K_d] and not keys_pressed[pygame.K_a]:
                    action = self.action_space[2]
                else:
                    action = self.action_space[1]
            else:
            # if not manual then send observation to agent
            # to get the agent's action
                if agent_input == 'Screenshot':
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
                    action_idx = get_action((self.pos, 
                                         (self.fruit[0][0], self.fruit[0][1]) if len(self.fruit) > 0 
                                         else None))
                    action = self.action_space[action_idx]
            
            # get next environment state
            reward = self.update_state(action)
            print(f'action={action}, reward={reward}')

            # send reward to agent
            if not manual:
                send_reward(reward)

            pygame.display.update()