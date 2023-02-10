import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from diff_cbf_qp import CBFQPLayer
from utils import soft_update, hard_update
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions,env,args):
        super(DeepQNetwork, self).__init__()

        self.env = env
        self.args = args
        

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=15000, eps_end=0.01, eps_dec=5e-4, replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', env=None, args=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_ctr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256,env=env,args=args)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        # CBF layer
        self.env = env
        self.cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
        self.diff_qp = args.diff_qp

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr%self.mem_size
        self.state_memory[index] = state.flatten()
        self.new_state_memory[index] = state_.flatten()
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state=np.asarray(observation)
            state = T.tensor(state).to(device=self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, memory, batch_size, updates, dynamics_model, memory_model=None, real_ratio=None):
        if self.mem_ctr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch+ self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        if memory_model and real_ratio:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=int(real_ratio*batch_size))
            state_batch_m, action_batch_m, reward_batch_m, next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m = memory_model.sample(
                batch_size=int((1-real_ratio) * batch_size))
            state_batch = np.vstack((state_batch, state_batch_m))
            action_batch = np.vstack((action_batch, action_batch_m))
            reward_batch = np.hstack((reward_batch, reward_batch_m))
            next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
            mask_batch = np.hstack((mask_batch, mask_batch_m))
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=batch_size)

        state_batch = T.FloatTensor(state_batch).to(self.device)
        next_state_batch = T.FloatTensor(next_state_batch).to(self.device)
        action_batch = T.FloatTensor(action_batch).to(self.device)
        reward_batch = T.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = T.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with T.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if self.diff_qp:  # Compute next safe actions using Differentiable CBF-QP
                next_state_action = self.get_safe_action(next_state_batch, next_state_action, dynamics_model)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi, _ = self.policy.sample(state_batch)
        if self.diff_qp:  # Compute safe action using Differentiable CBF-QP
            pi = self.get_safe_action(state_batch, pi, dynamics_model)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = T.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For Comet.ml logs
        else:
            alpha_loss = T.tensor(0.).to(self.device)
            alpha_tlogs =T.tensor(self.alpha)  # For Comet.ml logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
