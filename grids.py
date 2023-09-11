from class_environment import Environment
import torch
import torch.nn as nn

class Grid(Environment):
    def __init__(self, bounds, resolution, params, debug=False):
        super(Grid).__init__(bounds, resolution, params, debug)

        self.grid = torch.tensor([]).reshape(self.coords.shape)

        for i,value in enumerate(self.values):
            self.grid[i] = value
        
        self.current = [0, 0, 0, 0] #TODO insert random here 
    def return_pollution(self,index: tuple):
        return self.grid[index]
    

