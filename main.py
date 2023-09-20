# adapted from https://github.com/philtabor/Youtube-Code-Repository

import gymnasium as gym
from auv import AUV
from utils import plotLearning
import numpy as np
from env.grids import AUVGrid

# Hyperparameter
__length = 15
__network_input_size = ((__length*2)-1)**2 + 5
render_mode = "practical"
if __name__ == '__main__':
    env = AUVGrid(env_length=__length,render_mode=render_mode)#gym.make('gym_examples/GridWorld-v0')
    agent = AUV(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[__network_input_size], lr=0.001,eps_dec=0.05)
    scores, eps_history = [], []
    n_games = 50
    
    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()
        observation = np.concatenate((observation,np.array(info)))
        time_up = 0
        time_limit = int(len(env.env.values))/5
        while (not done) and time_up < time_limit:
            time_up+=1
            action = agent.choose_action(observation)
            observation_, reward, done, info_ = env.step(action,i>n_games-5)
            penalty = sum(info_[np.where(info_[:-5]< 0.0)])
            
            reward = reward * (agent.gamma**(time_up/6)) + done*100  + penalty # discount reward
            score += reward
            
            observation_ = np.concatenate((observation_
                                           ,np.array(info_)))
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            info=info_
        
        agent.epsilon = agent.epsilon - agent.eps_dec \
            if agent.epsilon > agent.eps_min else agent.eps_min
        scores.append(score/time_up)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i,'time_up ',time_up, 'score %.2f' % (score/time_up),
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'auv.png'
    plotLearning(x, scores, eps_history, filename)

