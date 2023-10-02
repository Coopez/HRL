# adapted from https://github.com/philtabor/Youtube-Code-Repository

import gymnasium as gym
from auv import AUV
from utils import plotLearning, get_sample
import numpy as np
from env.grids import AUVGrid
# Hyperparameter
__length = 15
__network_input_size =7 #((__length*2)-1)**2 + 5
_guideline = 30.0
render_mode = "human"
if __name__ == '__main__':
    env = AUVGrid(env_length=__length,render_mode=render_mode)#gym.make('gym_examples/GridWorld-v0')
    agent = AUV(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[__network_input_size], lr=0.001,eps_dec=0.01)
    scores, eps_history = [], []
    n_games = 120
    change_counter= 0
    old_score = 0.1
    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()
        observation = np.concatenate((observation,np.array(info),np.array([0.0])))
        time_up = 0
        time_limit = int(len(env.env.values))/2
        avg_loss = 0.0
        loss = []
        while (not done) and time_up < time_limit:
            s_ratio = time_up/time_limit
            action = agent.choose_action(observation)
            observation_, reward, done, info_ = env.step(action,i>n_games-5)
            #print(action,observation_)
            #penalty = sum(info_[np.where(info_[:-5]< 0.0)])
            exploration = env.observation_space[*env._agent_location]
            reward = (reward + done*100  + exploration )  * (agent.gamma**(time_up/6)) #+ penalty/10.0 # discount reward
            # if reward < 0.01:
            #     reward = 0.01
            #     change_counter+=1
            score += reward
            
            observation_ = np.concatenate((observation_
                                           ,np.array(info_),np.array([s_ratio])))
            # if reward > score/time_up or get_sample(0.35):
            agent.store_transition(observation, action, reward, 
                                        observation_, done)
            loss.append(agent.learn())

            # if change_counter >=5 and i<n_games/1.5 :
            #     agent.epsilon = min(0.9,agent.epsilon+0.0005)
            #     change_counter = 0
            # if old_score/time_up == score/time_up:
            #     change_counter+=1
            #old_score = score
            time_up+=1
            observation = observation_
            info=info_
            if  time_up % 40:
                agent.copy_weights()
        
        avg_loss = np.mean(agent.losses)
        agent.losses = []
        agent.epsilon = agent.epsilon - agent.eps_dec \
            if agent.epsilon > agent.eps_min else agent.eps_min
        scores.append(score/time_up)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i,'time_up ',time_up, 'score %.2f' % (score/time_up),
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon,
                'loss:  %.2f' % avg_loss)
    x = [i+1 for i in range(n_games)]
    filename = 'auv.png'
    plotLearning(x, scores, eps_history, filename)

