import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
torch.autograd.set_detect_anomaly(True)
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.sm = nn.Softmax()
        self.sig = nn.Sigmoid()
        self.relu= nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        state = self.sm(self.fc3(state))

        return state

class AUV():
    def __init__(self, gamma, epsilon, lr, input_dims,batch_size,n_actions, env_size,
                 max_mem_size=10000, eps_end=0.01,eps_dec=5e-4,layer_n1 = 256,layer_n2 = 256) -> None:

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min =eps_end
        self.eps_dec= eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.env_size = env_size
        # deep Q learning only for discrete action spaces- need diff alg if changing to a course based pathing approach 
        self.Q_eval = DeepQNetwork(self.lr,n_actions=n_actions,input_dims=input_dims,fc1_dims=layer_n1 ,fc2_dims=layer_n2 ) 
        self.Q_target = DeepQNetwork(self.lr,n_actions=n_actions,input_dims=input_dims,fc1_dims=layer_n1 ,fc2_dims=layer_n2 )
        #copy weights
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
        
        
        self.state_memory = np.zeros((self.mem_size,*input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_dims),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool_)
        
        self.losses= []
        # self.relevant_state_memory = np.zeros((self.mem_size,*input_dims),dtype=np.float32)
        # self.relevant_action_memory = np.zeros(self.mem_size,dtype=np.int32)
        # self.relevant_reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        # self.relevant_new_state_memory= np.zeros((self.mem_size,*input_dims),dtype=np.float32)
    def norm_loc(self,state): # minmax scaled now
        loc = state[0:2]
        loc[0] = (loc[0]-0.0) / ((self.env_size -1)-0.0)
        loc[1] = (loc[1]-0.0) / ((self.env_size -1)-0.0)
        state[0:2] = loc
        return state
    def store_transition(self,state,action,reward, state_,done):

        index=self.mem_cntr % self.mem_size # wrapping around to earliest memories if full
        self.state_memory[index] = self.norm_loc(state)
        self.new_state_memory[index] = self.norm_loc(state_)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # if reward > 90.0: 
        #     self.relevant_state_memory[index] = state
        #     self.relevant_action_memory[index] = action
        #     self.relevant_reward_memory[index] = reward
        #     self.new_state_memory[index] = state_
        self.mem_cntr += 1
    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation],dtype = torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            #action = torch.argmax(actions).item()
            action = np.random.choice([0,1,2,3], p=actions.detach().cpu().numpy().flatten())
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return # stop condition, apparently when memory counter samller than batch_size 
        
        self.Q_eval.optimizer.zero_grad()
        
        # probably just setting max for learning
        max_mem =  min(self.mem_cntr, self.mem_size)
        
        # random choice is for noobs
        batch = np.random.choice(max_mem,self.batch_size, replace= False) # also kind of an index
        
        # index = self.mem_cntr % self.mem_size
        # index_u = (self.mem_cntr - self.batch_size) % self.mem_size
        
        # if index < index_u:
        #     batch = np.concatenate((np.arange(index_u,self.mem_size),np.arange(0,index)))
        # else:
        #     batch = np.arange(index_u,index)

        batch_index = np.arange(self.batch_size,dtype=np.int32)
        # sending batches to network
        state_batch = torch.tensor(self.state_memory[batch],dtype=torch.float32).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
                self.new_state_memory[batch],dtype=torch.float32).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
                self.reward_memory[batch],dtype=torch.float32).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        # results
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        
        with torch.no_grad():    
            q_next = self.Q_target.forward(new_state_batch)
        
            q_next[terminal_batch] = torch.tensor(0.0).to(self.Q_target.device)

            q_target = reward_batch + torch.tensor(self.gamma).to(self.Q_target.device)*torch.max(q_next, dim=1)[0] # what is dis

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        with torch.no_grad():
            self.losses.append(loss.detach().cpu().numpy().tolist())
        
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        
     
        # control random choice here
        # self.epsilon = self.epsilon - self.eps_dec \
        #     if self.epsilon > self.eps_min else self.eps_min

    def copy_weights(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

# Legacy starts here

    #     #################
    #     self.pos = [0,0]
    #     self.states = [] # list of past states

    #     self.grid = grid # need link to class environment. I suggest 2D matrix in which index is grid position and value(s) are necessary sensory values
    #     self.grid_representation = None # it may also make sense to save a representation on what the auv actual knows of its environment - kind of like a filled in map

    #     # May also make sense to save important milestones, like the position of the highest concentration of pollution or all explored fields with pollution 
    #     self.source = None


    #     self.current_state = self.sense()

    # def sense(self):
    #     return self.grid.return_pollution(self.pos)

    # def move(self,action):
    #     """
    #     Make a move determined by the policy, save past state and sense current state
    #     """
    #     self.pos = self.pos + action
    #     self.states.append(self.current_state)
    #     self.current_state = self.sense()

