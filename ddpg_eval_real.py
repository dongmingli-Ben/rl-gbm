from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence
import os
from random import choice, randint
from copy import deepcopy
import pickle
import numpy as np
from util import execute_transactions, quantize
from random import random
import pandas as pd
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

    def __init__(self, utility_func, data: np.ndarray, num_of_days=1826, window_size=50) -> None:
        self.func = utility_func
        # history is the state ([bitcoin, gold, bitcoin prop, gold prop, portfolio value, remained days/1000])
        self.history = None
        self.day: int = None
        self.num = num_of_days
        self.window_size = window_size
        self.a_g = 1e-4
        self.a_b = 2e-4
        self.price = data
        self.price[:, 0] /= self.price[0, 0]
        self.price[:, 1] /= self.price[1, 1]  # first day is not trading day
        self.log_price = np.log(self.price)
        self.pre_gold_price = data[0, 1] if not np.isnan(data[0, 1]) else 0

    def set_state(self, obs):
        self.day = int(-1000*obs[-1][-1] - 1 + self.num)
        self.history = obs
        # pre_gold_price
        self.pre_gold_price = 0
        for i in range(self.day, -1, -1):
            if not np.isnan(self.price[i, 1]):
                self.pre_gold_price = self.price[i, 1]
                break

    def reset(self):
        # return obs
        self.day = 0
        self.history = [np.array([1., 1., 0., 0., 1., (self.num-self.day-1)/1000])]
        return np.stack(deepcopy(self.history))

    def step(self, action):
        # action is the weight of bitcoin, gold, and cash
        p_bitcoin, p_gold, prop_bitcoin, prop_gold, amount, days = self.history[-1].tolist()
        nxt_prop_bitcoin, nxt_prop_gold, nxt_prop_cash = quantize(action)

        if np.isnan(self.price[self.day][1]):
            # today is not a trading day
            # whether cash is enough
            if prop_gold - nxt_prop_gold > nxt_prop_cash:
                # cash not enough
                nxt_prop_bitcoin = 1 - nxt_prop_gold
            nxt_prop_gold = prop_gold
        nxt_prop_cash = 1 - nxt_prop_bitcoin - nxt_prop_gold

        pre_bitcoin_amount, pre_gold_amount, pre_cash_amount = \
            execute_transactions(prop_bitcoin, prop_gold,
                                 nxt_prop_bitcoin, nxt_prop_gold,
                                 amount, self.a_g, self.a_b, np.isnan(self.price[self.day][1]))


        nxt_bitcoin_amount = self.price[self.day+1][0]/self.price[self.day][0] \
                             * pre_bitcoin_amount
        if self.pre_gold_price == 0:
            # the first day of the period is not a trading day
            nxt_gold_amount = pre_gold_amount
            if not np.isnan(self.price[self.day][1]):
                self.pre_gold_price = self.price[self.day][1]
        elif np.isnan(self.price[self.day][1]):
            # today is not a trading day
            if np.isnan(self.price[self.day+1][1]):
                # tomorrow is not a trading day
                nxt_gold_amount = pre_gold_amount
            else:
                nxt_gold_amount = pre_gold_amount * \
                     self.price[self.day+1][1]/self.pre_gold_price
        else:
            # today is a trading day and the last available gold price is known
            if np.isnan(self.price[self.day+1][1]):
                # tomorrow is not a trading day
                nxt_gold_amount = pre_gold_amount
            else:
                nxt_gold_amount = self.price[self.day+1][1]/self.pre_gold_price \
                                    * pre_gold_amount
            self.pre_gold_price = self.price[self.day][1]
        nxt_cash_amount = pre_cash_amount
        nxt_amount = nxt_bitcoin_amount + nxt_gold_amount + nxt_cash_amount

        self.day += 1
        price = self.price[self.day].tolist()
        price[1] = self.pre_gold_price if np.isnan(price[1]) else price[1]
        obs = np.array(price + [nxt_prop_bitcoin, nxt_prop_gold, 
                                                          nxt_amount, (self.num-self.day-1)/1000])
        self.history.append(obs)
        # reward
        r = self.func(nxt_amount) - self.func(amount)
        return np.stack(self.history[-self.window_size:]), r, self.day == self.num - 1, {}

    def close(self):
        pass

def u_func(x):
    return 1000 * log(x)

def read_data(path, name):
    df = pd.read_csv(path)
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_raw_data():
    gold = read_data('data/LBMA-GOLD.csv', 'gold')
    bit = read_data('data/BCHAIN-MKPRU.csv', 'bitcoin')

    data = pd.merge(bit, gold, how='outer', on='date')
    return data

SEED = 73
data = load_raw_data()
DATA = data[['bitcoin', 'gold']].to_numpy()

def plot_eval(actor, ax, window_size, data, seed=SEED):
    np.random.seed(seed=seed)
    env = GBMEnvironment(utility_func=u_func, data=data, window_size=window_size)
    obs_ = env.reset()
    for i in range(env.num):
        obs_ = torch.tensor(obs_).cuda().float()
        action = actor(obs_.unsqueeze(0))
        obs_, r, done, info_ = env.step(action.squeeze().tolist())
        if done:
            break
    history = np.stack(env.history)
    # plt.subplot(211)
    ax[0].plot(history[:, 0], label='bitcoin')
    ax[0].plot(history[:, 1], label='gold')
    ax[0].plot(history[:, -2], label='portfolio')
    ax[0].legend(loc='upper right')
    # plt.show()

    # plt.subplot(212)
    ax[1].plot(history[:, 2], label='bitcoin weight')
    ax[1].plot(history[:, 3], label='gold weight')
    ax[1].legend(loc='upper right')
    # plt.show()

def visualize(path):
    num_step = int(re.search(r'step_(\d*).pt', path).group(1))
    checkpoint = torch.load(path)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    ax[0].cla()
    ax[1].cla()
    plot_eval(actor, ax, window_size, DATA)
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
# checkpoint = torch.load('results/ddpg/current_checkpoint_step_40000.pt')
# actor.load_state_dict(checkpoint['actor'])
# plot_eval(actor, ax, window_size, DATA, seed=SEED)
# plt.show()
ani = animation.FuncAnimation(fig, visualize, checkpoint_list, interval=1000)
ani.save(f'visualize-real-{SEED}.gif')