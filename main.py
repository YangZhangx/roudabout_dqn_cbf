import gym
import highway_env
import numpy as np
import dqn_agent
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

env = gym.make('roundabout-v0')
env.reset()
env.configure(config)
print(env.config)

agent = dqn_agent.Agent(gamma=0.99, epsilon=10, batch_size=32, n_actions=env.action_space.n, eps_end=0.01, input_dims=[350], lr=5e-4)
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
