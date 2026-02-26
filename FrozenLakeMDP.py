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

    def _state_to_pos(self, s):
        return divmod(s, self.grid_size)

    def _pos_to_state(self, row, col):
        return row * self.grid_size + col

    def _move(self, s, action):
        row, col = self._state_to_pos(s)
        
        if action == 0:  #Norte
            row = max(row - 1, 0)
        elif action == 1:  #Sur
            row = min(row + 1, self.grid_size - 1)
        elif action == 2:  #Este
            col = min(col + 1, self.grid_size - 1)
        elif action == 3:  #Oeste
            col = max(col - 1, 0)
        
        return self._pos_to_state(row, col)
    
    def _get_perpendicular_actions(self, action):
        if action in [0, 1]:  #Norte o Sur
            return [2, 3]     #Este, Oeste
        else:                 #Este u Oeste
            return [0, 1]     #Norte, Sur
    
        
    def _build_transition_matrix(self):
        for s in range(self.n_states):
            if s in self.holes or s == self.goal: #hoyos o ganar, acciones terminales
                for a in range(self.n_actions):
                    self.T[s, a, s] = 1.0
                continue

            for a in range(self.n_actions): #acción deseada
                intended_state = self._move(s, a)
                self.T[s, a, intended_state] += 1/3
                for perp_a in self._get_perpendicular_actions(a): #posibles acciones perpendiculares
                    perp_state = self._move(s, perp_a)
                    self.T[s, a, perp_state] += 1/3

    def _build_reward_matrix(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states): #s_prime es si llegamos a G
                    if s_prime == self.goal:
                        self.R[s, a, s_prime] = 1.0 #si llegamos nos da una recompensa
                    else:
                        self.R[s, a, s_prime] = 0.0
