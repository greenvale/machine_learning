import numpy as np
import scipy

def encode_state(glove_pos, fruit_pos_list) -> str:
    result = str(glove_pos)
    if fruit_pos_list:
        result += '--' + str(fruit_pos_list[0]) + ',' + str(fruit_pos_list[1])
    elif fruit_pos_list == None:
        result += '--None'
    return result

class Agent:
    def __init__(self, grid_size, glove_width):
        # 3 available actions to agent for each state
        self.num_actions = 3
        self.eps = 0.1
        self.alpha = 0.01
        self.gamma = 0.5

        # initialise Q table with random weights
        all_fruit_pos_list = [ (r,c) for r in range(grid_size) for c in range(grid_size) ] + [None]
        self.Q_table = { encode_state(glove_pos, fruit_pos_list) : np.random.randn(self.num_actions) 
                        for glove_pos in range(grid_size - glove_width + 1) 
                        for fruit_pos_list in all_fruit_pos_list }

        # initialise keys for cache
        self.cache = {
            'prev_state':None,
            'new_state':None,
            'action':None,
            'reward':None
        }
    
    def take_action(self, state):
        
        state_encoded = encode_state(*state)
        print(f'State={state_encoded}')

        self.cache['new_state'] = state_encoded

        # now have new state for prev action
        # in cache: state_(t-1), action_(t-1), reward_(t), state_(t)
        
        # now update Q table
        # Q_table[old_state, action] = (1-alpha)*Q_table[old_state, action] + alpha*(reward + gamma*sum(Q_table[new_state, actions]))
        if self.cache['reward'] != None and self.cache['action'] and self.cache['prev_state'] and self.cache['new_state']:
            td_target = self.cache['reward'] + self.gamma*self.Q_table[self.cache['new_state']].max()
            print("td_target=",td_target)
            self.Q_table[self.cache['prev_state']][self.cache['action']] *= 1.0 - self.alpha
            self.Q_table[self.cache['prev_state']][self.cache['action']] += self.alpha * td_target

        ## ------------------------------

        # get action from epsilon-greedy policy
        policy_probs = np.ones((self.num_actions,)) * (self.eps/self.num_actions)
        policy_probs[self.Q_table[state_encoded].argmax()] += 1.0 - self.eps
        rv = scipy.stats.multinomial(1, policy_probs)
        action_idx = rv.rvs(1).argmax()

        # send state_(t) to state_(t-1)
        # cache action
        self.cache['prev_state'] = self.cache['new_state']
        self.cache['action'] = action_idx
        self.cache['new_state'] = None
        self.cache['reward'] = None

        return action_idx

    def receive_reward(self, reward):
        # get reward --> send to reward_(t) in cache
        self.cache['reward'] = reward