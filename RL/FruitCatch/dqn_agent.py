import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from collections import deque

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(324, 3)

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.lin1(x)
        if y is not None:
            loss = F.mse_loss(x, y)
            return x[0], loss
        else:
            return x[0]

# copy the parameter values from src model to dest model
# so dest model has same parameter values as src model
def copy_params(model_src, model_dest):
    for ps,pd in zip(model_src.parameters(), model_dest.parameters()):
        pd.data = ps.data

class Transition:
    action_idx = None       # action @ t
    state = None            # state @ t
    reward = None           # reward @ t+1
    future_state = None     # state @ t+1

# Every transition gets added to the replay buffer
# Batch is sampled randomly from the replay buffer
# The target network uses the old weights
# The predictor network's Q values are sampled at each timestep

# @ state s, the agent samples predictor network to get Q values
# it then chooses action a from epsilon greedy policy
# given reward r and new state s'
# This gives Q*(s, a; theta_i) = r + gamma * max_{a'} ( Q*(s', a'; theta_[i-1]) )
# For true action-value function Q*

# So the TD target for transition {s,a,r,s'} is TD_{s,a,r,s'} = r + gamma * max_{a'} ( Q*(s', a'; theta_[i-1]) )
# Our error is therefore Q(s, a; theta_i) - TD_{s,a,r,s'}
# After N steps of collecting experience in the replay buffer, we sample a batch and perform SGD on the predictor model
# We then swap the weights from the predictor model into the target model

class DQN_Agent:
    def __init__(self, grid_size, glove_width):
        # 3 available actions to agent for each state
        self.grid_size = grid_size
        self.glove_width = glove_width
        self.num_actions = 3
        self.action_space = [-1, 0, 1]
        self.eps = 0.1
        self.eps_decay = 0.5
        self.alpha = 0.1
        self.gamma = 0.8
        self.training = True
        self.learning_rate = 0.01

        self.replay_buffer_size = 1000
        self.replay_buffer = deque([], maxlen=self.replay_buffer_size)

        self.batch_size = 32
        self.transfer_freq = 100

        self.predictor_model = Model()
        self.target_model = Model()
        copy_params(self.predictor_model, self.target_model)

    def train_predictor(self):
        # choose a random array of transitions from the replay buffer
        # construct the X inputs and y targets from these transitions to get a batch
        # Adjust the weights of the predictor model
        batch_data = torch.zeros((self.batch_size, 2, self.grid_size, self.grid_size), dtype=torch.float)
        batch_targets = torch.zeros((self.batch_size, 3), dtype=torch.float)
        idx = torch.randint(0, len(self.replay_buffer) - 1, (self.batch_size,))
        bi = 0
        for i in idx.tolist():
            trans = self.replay_buffer[i]
            batch_data[bi, :, :, :] = trans.state

            # get the TD target value of the Q(s, a) value (a'th element of output Q(s))
            td_target = trans.reward + self.gamma*(self.target_model(trans.future_state.unsqueeze(0))).max()
            
            # target is then a zero vector except for the a'th value which is the TD target
            batch_targets[bi, trans.action_idx] = td_target

            bi += 1
        
        # do a forward pass using the batch
        pred, loss = self.predictor_model(batch_data, batch_targets)

        # do backward propagation on the predictor model weights
        for p in self.predictor_model.parameters():
            p.grad = None
        loss.backward()
        for p in self.predictor_model.parameters():
            p.data += -self.learning_rate * p.grad

    # transfer the predictor model weights to the target model
    def iterate_model(self):
        copy_params(self.predictor_model, self.target_model)

    # decide action given state using the epsilon-greedy policy
    def eval_policy(self, state):
        policy_probs = np.ones((self.num_actions,)) * (self.eps/self.num_actions)

        policy_probs[ self.predictor_model(state.unsqueeze(0)).argmax()] += 1.0 - self.eps
        
        rv = scipy.stats.multinomial(1, policy_probs)
        action_idx = rv.rvs(1).argmax()

        return action_idx
    
    # the agent receives new state information
    # if this is the first timestep then open a new transition
    # otherwise complete the existing transition by given future_state
    # then open a new transition for the state
    def take_action(self, state):
        if len(self.replay_buffer) == 0:
            self.replay_buffer.append(Transition())
            self.replay_buffer[-1].state = state
        else:
            assert(self.replay_buffer[-1].reward is not None) ## reward must have been provided to agent from self.receive_reward()
            self.replay_buffer[-1].future_state = state
            self.replay_buffer.append(Transition())
            self.replay_buffer[-1].state = state

        action_idx = self.eval_policy(state)

        action = self.action_space[action_idx]

        self.replay_buffer[-1].action_idx = action_idx

        return action

    # receives reward from environment and stores in cache
    def receive_reward(self, reward):
        self.replay_buffer[-1].reward = reward

    def drop_eps(self):
        self.eps *= self.eps_decay