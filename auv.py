import torch
import random



class AUV():
    def __init__(self,grid, actions) -> None:
    
        self.pos = [0,0]
        self.states = [] # list of past states

        self.grid = grid # need link to class environment. I suggest 2D matrix in which index is grid position and value(s) are necessary sensory values
        self.grid_representation = None # it may also make sense to save a representation on what the auv actual knows of its environment - kind of like a filled in map

        # May also make sense to save important milestones, like the position of the highest concentration of pollution or all explored fields with pollution 
        self.source = None

        self.ActionSpace = actions

        self.current_state = self.sense()

    def sense(self):
        return self.grid.return_pollution(self.pos)

    def move(self,action):
        """
        Make a move determined by the policy, save past state and sense current state
        """
        self.pos = self.pos + self.ActionSpace[action]
        self.states.append(self.current_state)
        self.current_state = self.sense()

