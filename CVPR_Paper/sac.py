import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
import csv
torch.manual_seed(11)
#Hyperparameters
lr_pi           = 0.00005
lr_q            = 0.0001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 64
buffer_limit    = 50000
tau             = 0.001 # for target network soft update
target_entropy  = -2.0 # for automated alpha update
lr_alpha        = 0.0001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(35, 256)
        self.fc_mu = nn.Linear(256,1)
        self.fc_std  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
#         std = F.softplus(self.fc_std(x))
        std = abs(F.softplus(self.fc_std(x)))
#         print(mu,std)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(35, 128)
        self.fc_a = nn.Linear(1,128)
        self.fc_cat = nn.Linear(256,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

def get_reward_costant_k1(state, action, next_state, done, k1, k2, action_amount, i_episode):
    #if done, return Yield (topwt) - k*cost
    penalty = 0
    #k1=10**(-3)*(i_episode*2)
    # base cost is current input action
    if action_amount > 160:
        if action!=0:
            penalty = (action_amount - 160)
    if done:
        reward = state[29] - k2*action - k1*penalty
        return reward
    #if done, return Yield (topwt) - k*cost
    else:
        reward = -k2*(action + state[25]) - k1*penalty
        return reward
    # otherwise, return -k*(action+leaching)


def dict2array(state):
    new_state = []
    for key  in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])        
    state = np.asarray(new_state)
    return state


def main():
#     env = gym.make('Pendulum-v0')

    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',  # assuming (modified) DSSAT has been installed in /opt/dssat_pdi
        'log_saving_path': './logs/dssat-pdi.log',  # if you want to save DSSAT outputs for inspection
        # 'mode': 'irrigation',  # you can choose one of those 3 modes
        # 'mode': 'fertilization',
        'mode': 'all',
        'seed': 123456,
        'random_weather': False,  # if you want stochastic weather
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20
    to_save = []
    for n_epi in range(1500+1):
        action_amount = 0
        s = dict2array(env.reset())
        done = False

        while not done:
            a, log_prob= pi(torch.from_numpy(s).float())
#             print('action',a)
            scaled_a = int((a+1)*100)
            
            action_amount = action_amount + scaled_a
            a = {
                'anfer': scaled_a,  # if mode == fertilization or mode == all ; nitrogen to fertilize in kg/ha
                'amir': 0,  # if mode == irrigation or mode == all ; water to irrigate in L/ha
            }
            s_prime, r, done, info = env.step(a)
            if done:
                s_prime = np.zeros(35)
                reward = get_reward_costant_k1(s, a['anfer'], s_prime, done, 0.5, 0.1, action_amount, n_epi)
                memory.put((s, a['anfer'], reward, s_prime, done))
                score += reward
                count = 0   
                break
  #400-600 mm water 

            s_prime = dict2array(s_prime)  
            r = get_reward_costant_k1(s, a['anfer'], s_prime, done, 0.5, 0.1, action_amount, n_epi)
            memory.put((s, a['anfer'], r, s_prime, done))
            score +=r
            s = s_prime
                
        if memory.size()>1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)
                
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, (score/print_interval).item(), pi.log_alpha.exp()))
            to_save.append((score/print_interval).item())
            # print(to_save)
            score = 0.0

            if len(to_save)!=0 and max(to_save)>=500:
                with open('TEMP', 'w') as f:
                    
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    write.writerow(to_save)
    env.close()

if __name__ == '__main__':
    main()