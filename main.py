import gym
import highway_env
import numpy as np
import dqn_agent
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.observation import ObservationType

config = {
'action': {'type': 'DiscreteMetaAction'},
'centering_position': [0.3, 0.5],
'collision_reward': -1000,
'controlled_vehicles': 1,
'duration': 10,
'high_speed_reward': 5,
'lane_change_reward': 2,
'lanes_count': 2,
'observation': {'absolute': False,
                'features': ['presence',
                            'x',
                            'y',
                            'vx',
                            'vy',
                            'cos_h',
                            'sin_h'],
                'features_range': {'vx': [-10, 10],
                                'vy': [-10, 10],
                                'x': [-200, 200],
                                'y': [-200, 200]},
                'normalize': True,
                'order': 'sorted',
                'type': 'Kinematics',
                'vehicles_count': 50},
'vehicles_count': 50,
'reward_speed_range': [15, 25],
'right_lane_reward': 0,
}

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# Environment Args
parser.add_argument('--env-name', default="Unicycle", help='Options is Unicycle')
# Comet ML
parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
parser.add_argument('--comet_key', default='', help='Comet API key')
parser.add_argument('--comet_workspace', default='', help='Comet workspace')
# SAC Args
parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only in available test mode')
parser.add_argument('--output', default='output', type=str, help='')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 5 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=12345, metavar='N',
                    help='random seed (default: 12345)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--max_episodes', type=int, default=200, metavar='N',
                    help='maximum number of episodes (default: 200)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
# CBF, Dynamics, Env Args
parser.add_argument('--no_diff_qp', action='store_false', dest='diff_qp', help='Should the agent diff through the CBF?')
parser.add_argument('--gp_model_size', default=3000, type=int, help='gp')
parser.add_argument('--gp_max_episodes', default=100, type=int, help='gp max train episodes.')
parser.add_argument('--k_d', default=3.0, type=float)
parser.add_argument('--gamma_b', default=20, type=float)
parser.add_argument('--l_p', default=0.03, type=float,
                    help="Look-ahead distance for unicycle dynamics output.")
# Model Based Learning
parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
# Compensator
parser.add_argument('--comp_rate', default=0.005, type=float, help='Compensator learning rate')
parser.add_argument('--comp_train_episodes', default=200, type=int, help='Number of initial episodes to train compensator for.')
parser.add_argument('--comp_update_episode', default=50, type=int, help='Modulo for compensator updates')
parser.add_argument('--use_comp', type=bool, default=False, help='Should the compensator be used.')
args = parser.parse_args()

env = gym.make('roundabout-v0')
env.reset()
env.configure(config)
print(env.config)

agent = dqn_agent.Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=env.action_space.n, eps_end=0.01, input_dims=[350], lr=5e-4, eps_dec=1e-5, env_name='roundabout-v0', chkpt_dir='models/')
scores, eps_history = [], []
n_games = 500
for i in range(n_games):
    score = 0
    terminated = False
    truncated = False
    observation = env.reset()
    observation=observation[0]
    while not (terminated or truncated):
        env.render()
        action = agent.choose_action(observation.flatten())
        observation_, reward, terminated, truncated, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, (terminated or truncated))
        agent.learn()
        observation = observation_
        

    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    print(f'episode: {i} | score: {score} | average score: {avg_score} | epsilon: {agent.epsilon}')
    env.close()
