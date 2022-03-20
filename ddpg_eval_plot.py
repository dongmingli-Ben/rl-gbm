from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence
import os
from random import randint
from copy import deepcopy
import pickle
import numpy as np
from util import execute_transactions, quantize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

class Actor(nn.Module):

    def __init__(self, hidden_size):
        super(Actor, self).__init__()
        self.hidden = hidden_size
        self.lstm = nn.LSTM(input_size=6,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, return_logits=False):
        x = x.float()
        bs = x.batch_sizes[0].item() if isinstance(x, PackedSequence) else x.size(0)
        h0 = torch.zeros((1, bs, self.hidden)).to(x[0])
        c0 = torch.zeros((1, bs, self.hidden)).to(x[0])
        output, (hn, cn) = self.lstm(x, (h0, c0))
        if isinstance(x, PackedSequence):
            padded_output, seq_len = pad_packed_sequence(output, batch_first=True)
            index = seq_len.to(padded_output.device).reshape(-1, 1)-1
            index = index.unsqueeze(-1).expand(-1, -1, padded_output.size(-1))
            hidden = torch.gather(padded_output, 1, index).squeeze()
        else:
            hidden = output[:, -1]
        out = self.proj(hidden)
        if return_logits:
            return out
        return F.softmax(out, dim=-1)


class GBMEnvironment:

    def __init__(self, utility_func, num_of_days=1826, window_size=50, eval=False) -> None:
        self.func = utility_func
        # history is the state ([bitcoin, gold, bitcoin prop, gold prop, portfolio value, remained days/1000])
        self.history = None
        self.day: int = None
        self.num = num_of_days
        self.window_size = window_size
        self.a_g = 1e-4
        self.a_b = 2e-4
        self.i = 1
        self.eval = eval

    def reset(self, with_mean_cov=False):
        # return obs
        mu = np.random.normal(5e-4, 5e-4, 2)
        cov = np.diag(np.random.normal(1e-4, 5e-5, 2)) + np.random.normal(5e-5, 1e-5, (2, 2))
        cov = np.abs((cov + cov.T) / 2)
        factor = max(self.i, 100000) / 100000 if not self.eval else 1
        cov = cov * factor
        # generate the GBM
        bm = np.zeros((self.num, 2))
        w = np.random.multivariate_normal(mu, cov, self.num)
        for i in range(self.num-1):
            bm[i+1] = bm[i] + w[i]
        # GBM
        gbm = np.exp(bm)
        self.price = gbm
        self.log_price = bm

        self.day = 0
        self.history = [np.array([1., 1., 0., 0., 1., (self.num-self.day-1)/1000])]
        print(f'mean: {mu}\ncov: \n{cov}')
        self.i += 1
        if with_mean_cov:
            return np.stack(deepcopy(self.history)), mu, cov
        return np.stack(deepcopy(self.history))

    def step(self, action):
        # action is the weight of bitcoin, gold, and cash
        p_bitcoin, p_gold, prop_bitcoin, prop_gold, amount, days = self.history[-1].tolist()
        nxt_prop_bitcoin, nxt_prop_gold, nxt_prop_cash = quantize(action)
        pre_bitcoin_amount, pre_gold_amount, pre_cash_amount = \
            execute_transactions(prop_bitcoin, prop_gold,
                                 nxt_prop_bitcoin, nxt_prop_gold,
                                 amount, self.a_g, self.a_b, 0)
        nxt_amount = pre_bitcoin_amount*self.price[self.day+1][0]/self.price[self.day][0] \
            + pre_gold_amount*self.price[self.day+1][1]/self.price[self.day][1] \
            + pre_cash_amount
        self.day += 1
        obs = np.array(self.price[self.day].tolist() + [nxt_prop_bitcoin, nxt_prop_gold, 
                                                          nxt_amount, (self.num-self.day-1)/1000])
        self.history.append(obs)
        # reward
        r = self.func(nxt_amount) - self.func(amount)
        return np.stack(self.history[-self.window_size:]), r, self.day == self.num - 1, {}

    def close(self):
        pass

def u_func(x):
    return 1000 * log(x)

SEED = 73

def plot_eval(actor, ax, window_size, seed=SEED):
    np.random.seed(seed=seed)
    env = GBMEnvironment(utility_func=u_func, window_size=window_size, eval=True)
    obs_ = env.reset()
    for i in range(env.num):
        obs_ = torch.tensor(obs_).cuda().float()
        action = actor(obs_.unsqueeze(0))
        obs_, r, done, info_ = env.step(action.squeeze().tolist())
        if done:
            break
    history = np.stack(env.history)
    # plt.subplot(211)
    ax[0].plot(history[:, 0], label='asset 1')
    ax[0].plot(history[:, 1], label='asset 2')
    ax[0].plot(history[:, -2], label='portfolio')
    ax[0].legend(loc='upper right')
    # plt.show()

    # plt.subplot(212)
    ax[1].plot(history[:, 2], label='asset 1 weight')
    ax[1].plot(history[:, 3], label='asset 2 weight')
    ax[1].legend(loc='upper right')
    # plt.show()

def visualize(path):
    num_step = int(re.search(r'step_(\d*).pt', path).group(1))
    checkpoint = torch.load(path)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    ax[0].cla()
    ax[1].cla()
    plot_eval(actor, ax, window_size)
    ax[0].set_title(f'Step num {num_step}')

actor = Actor(64).cuda()
window_size = 100

checkpoint_list = []
step_nums = []
for dirpath, dirnames, files in os.walk('results/ddpg'):
    for file in files:
        if file.endswith('.pt'):
            path = os.path.join(dirpath, file)
            checkpoint_list.append(path)
            step_nums.append(int(re.search(r'step_(\d*).pt', path).group(1)))
# sort
checkpoint_list, step_nums = list(zip(*sorted(zip(checkpoint_list, step_nums), key=lambda t: t[1])))

records = []
print(f'Visualizing for {len(checkpoint_list)} models')
fig, ax = plt.subplots(nrows=2, ncols=1)

ani = animation.FuncAnimation(fig, visualize, checkpoint_list, interval=1000)
ani.save(f'visualize-{SEED}.gif')