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

class QNet(nn.Module):

    def __init__(self, hidden_size):
        super(QNet, self).__init__()
        self.hidden = hidden_size
        self.lstm = nn.LSTM(input_size=6,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size+3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, obs, action):
        obs = obs.float()
        bs = obs.batch_sizes[0].item() if isinstance(obs, PackedSequence) else obs.size(0)
        h0 = torch.zeros((1, bs, self.hidden)).to(obs[0])
        c0 = torch.zeros((1, bs, self.hidden)).to(obs[0])
        output, (hn, cn) = self.lstm(obs, (h0, c0))
        if isinstance(obs, PackedSequence):
            padded_output, seq_len = pad_packed_sequence(output, batch_first=True)
            index = seq_len.to(padded_output.device).reshape(-1, 1)-1
            index = index.unsqueeze(-1).expand(-1, -1, padded_output.size(-1))
            hidden = torch.gather(padded_output, 1, index).squeeze()
        else:
            hidden = output[:, -1]
        x = torch.cat([hidden, action],dim=-1)
        out = self.net(x)
        return out

def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def release(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True

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

@torch.no_grad()
def update_target(target: nn.Module, model: nn.Module, polyak):
    for p_t, p in zip(target.parameters(), model.parameters()):
        p_t.mul_(polyak)
        p_t.add_((1-polyak)*p)

def train(actor, qnet, actor_optimizer, q_optimizer, max_steps=500, log_period=50, save_dir='save', 
        update_time=1, update_batch=1, max_buffer_size=10000, batch_size=4, gamma=0.999, 
        start_after_steps=1000, polyak=0.995, early_stopping=10, save_period=10000,
        func=log, window_size=50):
    actor_t = deepcopy(actor)
    qnet_t = deepcopy(qnet)
    env = GBMEnvironment(utility_func=func, window_size=window_size)
    eval_env = GBMEnvironment(utility_func=func, window_size=window_size, eval=True)
    # replay buffer [obs, action, reward, next obs, done]
    buffer = []
    # 10x return as win
    info = {'rewards': [], 'first_win': None, 'avg_q': []}
    obs = env.reset()
    obs = torch.tensor(obs).cuda()
    total_loss_q, total_loss_actor = 0, 0
    win = 0
    for step in range(max_steps):
        with torch.no_grad():
            action = actor(obs.unsqueeze(0), True) + .5*torch.randn(3).cuda()
            action = F.softmax(action, dim=-1).squeeze()
        next_obs, reward, done, info_ = env.step(action.tolist())
        next_obs = torch.tensor(next_obs).cuda()
        if len(buffer) > max_buffer_size:
            # index = randint(0, max_buffer_size-1)
            # buffer.pop(index)
            buffer = buffer[-max_buffer_size+1:]
        buffer.append((obs.cpu(), action.cpu(), reward, next_obs.cpu(), done))
        obs = next_obs
        if done:
            obs = env.reset()
            obs = torch.tensor(obs).cuda()
        if step < start_after_steps: continue
        if (step+1) % update_time == 0:
            actor.train()
            qnet.train()
            actor_t.train()
            qnet_t.train()
            # import pdb; pdb.set_trace()
            for num in range(update_batch):
                batches = []
                for i in range(batch_size):
                    batches.append(buffer[randint(0, len(buffer)-1)])
                obs_, action, reward, next_obs, done = zip(*batches)
                obs_ = pack_sequence(obs_, enforce_sorted=False).cuda()
                action = torch.stack(action).cuda()
                reward = torch.tensor(reward).cuda()
                next_obs = pack_sequence(next_obs, enforce_sorted=False).cuda()
                done = torch.tensor(done).cuda().float()

                with torch.no_grad():
                    target_q = reward + gamma*(1-done)*qnet_t(next_obs, actor_t(next_obs)).squeeze(-1)
                q_value = qnet(obs_, action).squeeze(-1)
                loss_q = F.mse_loss(q_value, target_q)
                total_loss_q += loss_q.item()
                q_optimizer.zero_grad()
                loss_q.backward()
                q_optimizer.step()

                freeze(qnet)
                loss_actor = - qnet(obs_, actor(obs_)).mean()
                total_loss_actor += loss_actor.item()
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()
                release(qnet)

                update_target(actor_t, actor, polyak)
                update_target(qnet_t, qnet, polyak)

        # save model
        if (step+1) % save_period == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.path.join(save_dir, f'current_checkpoint_step_{step+1}.pt')
            torch.save({
                'actor': actor.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'qnet': qnet.state_dict(),
                'q_optimizer': q_optimizer.state_dict(),
            }, path)
            path = os.path.join(save_dir, 'info.txt')
            with open(path, 'wb') as f:
                pickle.dump(info, f)


        with torch.no_grad():
            if (step+1)%log_period == 0:
                actor.eval()
                qnet.eval()
                actor_t.eval()
                qnet_t.eval()
                reward = 0
                total_q = 0
                obs_ = eval_env.reset()
                for i in range(eval_env.num):
                    obs_ = torch.tensor(obs_).cuda().float()
                    action = actor(obs_.unsqueeze(0))
                    total_q += qnet(obs_.unsqueeze(0), action).item()
                    obs_, r, done, info_ = eval_env.step(action.squeeze().tolist())
                    reward += r * gamma**i
                    if done:
                        break
                print('step', '%6d'%(step+1), 'reward', '%3.2f'%reward, 
                    'terminal net value', '%3.2f'%obs_[-1][-2], 'total steps', '%3d'%(i+1),
                    'Avg actor loss', round(total_loss_actor/log_period, 3),
                    'Avg q loss', round(total_loss_q/log_period, 3),
                    'Avg q value', round(total_q/(i+1), 3))
                total_loss_q, total_loss_actor = 0, 0
                info['rewards'].append(reward)
                info['avg_q'].append(total_q/(i+1))
                if info['first_win'] is None and obs_[-1][-1] > 10:
                    info['first_win'] = step + 1
                if obs_[-1][-1] > 10:
                    win += 1
                else:
                    win = 0
                if win > early_stopping: break

    eval_env.close()
    env.close()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, 'current_checkpoint.pt')
    torch.save({
        'actor': actor.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'qnet': qnet.state_dict(),
        'q_optimizer': q_optimizer.state_dict(),
    }, path)
    path = os.path.join(save_dir, 'info.txt')
    with open(path, 'wb') as f:
        pickle.dump(info, f)
    print(info['first_win'])

def u_func(x):
    return 1000 * log(x)

def sharpe(x):
    x = x[1:] / x[:-1] - 1
    m = x.mean()
    std = x.std()
    return m / std

def standard_deviation(x):
    x = x[1:] / x[:-1] - 1
    # m = x.mean()
    std = x.std()
    return std

def get_metrics(history, mean):
    # average proportion holding the highest mean return asset
    # ratio of net value against the asset with highest mean return
    # diff of the reward against the asset with highest mean return
    # ratio of sharpe against the asset with highest mean return
    # ratio of std against the asset with highest mean return
    # std of the proportion of asset 1
    if mean.max() > 0:
        idx = np.argmax(mean)
        prop = history[:, 2+idx].mean()
        value_ratio = history[-1, -2] / history[-1, idx]
        reward_diff = u_func(history[-1, -2]) - u_func(history[-1, idx])
        sharpe_ratio = sharpe(history[:, -2]) / sharpe(history[:, 2+idx])
        std_ratio = standard_deviation(history[:, -2]) / standard_deviation(history[:, 2+idx])
    else:
        # cash prop
        prop = 1 - history[:, 2].mean() - history[:, 3].mean()
        value_ratio = history[-1, -2]
        reward_diff = u_func(history[-1, -2]) - u_func(1)
        sharpe_ratio = 0
        std_ratio = np.inf
    prop_std = np.std(history[:, 2])

    metrics = {
        'prop': prop,
        'value_ratio': value_ratio,
        'reward_diff': reward_diff,
        'sharpe_ratio': sharpe_ratio,
        'std_ratio': std_ratio,
        'prop_std': prop_std,
    }
    return metrics

@torch.no_grad()
def evaluate(actor, num, window_size):
    env = GBMEnvironment(utility_func=u_func, window_size=window_size, eval=True)
    results = {}
    for i in range(num):
        obs_, mu, cov = env.reset(with_mean_cov=True)
        for j in range(env.num):
            obs_ = torch.tensor(obs_).cuda().float()
            action = actor(obs_.unsqueeze(0))
            obs_, r, done, info_ = env.step(action.squeeze().tolist())
            if done:
                break
        history = env.history
        metrics = get_metrics(history, mu)
        for key, val in metrics.items():
            results[key] = results.get(key, []) + [val]
    for key, val in results.items():
        results[key] = np.mean(val)
    return results

actor = Actor(64).cuda()
qnet = QNet(64).cuda()
checkpoint = torch.load('results/ddpg/current_checkpoint_step_40000.pt')
actor.load_state_dict(checkpoint['actor'])
qnet.load_state_dict(checkpoint['qnet'])
actor.eval()
qnet.eval()

env = GBMEnvironment(utility_func=u_func, window_size=100, eval=True)
reward = 0
total_q = 0
obs_ = env.reset()
for i in range(env.num):
    obs_ = torch.tensor(obs_).cuda().float()
    action = actor(obs_.unsqueeze(0))
    total_q += qnet(obs_.unsqueeze(0), action).item()
    obs_, r, done, info_ = env.step(action.squeeze().tolist())
    reward += r
    if done:
        break

result = np.stack(env.history)
plt.subplot(211)
plt.plot(result[:, 0], label='asset 1')
plt.plot(result[:, 1], label='asset 2')
plt.plot(result[:, -2], label='portfolio')
plt.legend(loc='upper right')
# plt.show()

plt.subplot(212)
plt.plot(result[:, 2], label='asset 1 weight')
plt.plot(result[:, 3], label='asset 2 weight')
plt.legend(loc='upper right')
plt.show()