import numpy as np

class FrozeLakeMDP:
    def __init__(self):
        self.n_states = 16
        self.n_actions = 4  # 0:Norte, 1:Sur, 2:Este, 3:Oeste
        
        self.grid_size = 4
        
        #mapa 4x4
        # S F F F
        # F H F H
        # F F F H
        # H F F G
        
        self.holes = [5, 7, 11, 12]
        self.goal = 15
        
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        self._build_transition_matrix()
        self._build_reward_matrix()

    #def _state_to_pos():

    #def _pos_to_state():

    #def _move():

    