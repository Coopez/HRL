from auv import AUV
from env.class_environment import Environment

def islegal(action,auv,grid):
    location = auv.pos
    bound_x, bound_y = grid.bounds 
    post = location + action # assuming [x,y]
    if post[0] < bound_x[0] or post[0] > bound_x[1]: 
        return False
    if post[1] < bound_y[0] or post[1] > bound_y[1]:
        return False
    return True

def policy(grid,auv,model):
    """
    makes decision about legal- and best action
    
    
    """
    actionspace = auv.ActionSpace
    legal_actions = []
    for action in actionspace:

        # check if legal
        if islegal(action,auv,grid):
            legal_actions.append(action)

    best_action = model(legal_actions)
    return best_action

def reward(action,auv):
    v = 0
    cs = auv.current_state
    pc = auv.states
    
    
    
    return v
def main():
    # without doing much here
    actionSpace = {
            "right": [1,0], # move right
            "left": [-1,0], # move left
            "up": [0,1], #
            "down": [1,0] #
        }
    map = Environment()
    auv = AUV(map,actionSpace)
    model = None # need some NN here
    done = False
    steps = 0
    while not done:
        
        action = policy(map,auv,model,actionSpace)
        reward = reward(action,auv)
        model.update(reward)
        auv.move()
        steps += 1
        done = steps > 1000