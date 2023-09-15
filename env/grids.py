from env.class_environment import Environment
import torch
import torch.nn as nn
import numpy as np
import math
# class Grid(Environment):
#     def __init__(self, bounds, resolution, params, debug=False):
#         super(Grid,self).__init__(bounds, resolution, params, debug)

#         self.grid = torch.tensor([]).reshape(self.coords.shape)

#         for i,value in enumerate(self.values):
#             self.grid[i] = value
        
#         self.current = [0, 0, 0, 0] #TODO insert random here 
#     def return_pollution(self,index: tuple):
#         return self.grid[index]
    

from functools import reduce

# gym based env try. 
# Need gymnasium as python 3.11 installed. Not sure if I could donwgrade python given other dependencies
import gymnasium as gym
from gymnasium import spaces

 # if doubble inheritance doesnt work, just pass the class as input
class AUVGrid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,env_length,render_mode=None):
        super(AUVGrid,self).__init__()
        if input==None:
            self.env = Environment.get_random_env(leak_len=env_length, type='elliptical')
        else:
            # implement explicit input here
            raise NotImplementedError
        # self.env_length = env_length
        # self.pollution = self.env.values.reshape((env_length*2,env_length*2))
        # self.cs = self.env.params['current_strength']
        # self.cdir = self.env.params['current_direction']

        # self.size = len(self.env_length) # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # pollution grid
        self.import_env()
        """
        importing Waruum env and assigning to local self values:
        self.env_length - Length of the grid
        self.pollution - Pollution values in a grid
        self.cs - Current strength
        self.cdir - current direction in degrees
        self.size # The size of the square grid
        """

        
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        xbound = self.bounds[0]
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(xbound[0], xbound[1], shape=(2,), dtype=int),
                #"target": spaces.Box(xbound[0], xbound[1], shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
 
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return {"agent": self._agent_location }#, "target": self._target_location}
   
    def _get_info(self):
        #return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        # this needs to be changed to pollution levels
        p = self.pollution[self._agent_location]
        cs = self.cs # can be expanded if current is uneven
        csdir = self.cdir 
        return [p,cs,csdir]
    
    def angle_to_vector(angle_degrees):
        # Convert degrees to radians
        angle_radians = math.radians(angle_degrees)
        
        # Calculate the x and y components of the unit vector
        x = math.cos(angle_radians)
        y = math.sin(angle_radians)
    
        return (x, y)
    
    def import_env(self,env_length):
        self.env_length = env_length
        self.pollution = self.env.values.reshape((env_length*2,env_length*2))
        self.cs = self.env.params['current_strength']
        self.cdir = self.env.params['current_direction']

        self.size = len(self.env_length) # The size of the square grid

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        if options == None:
            self.env = Environment.get_random_env(leak_len=self.env_length, type='elliptical')
            self.import_env(self.env_length)
        else:
            raise NotImplementedError
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        ### SO these are the rewards and termination condition
        terminated =  1 if self.pollution[self._agent_location]>0.5 else 0
        
        reward = self.pollution[self._agent_location] - self.cs*(np.matmul(self.angle_to_vector(self.cdir),direction))
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()