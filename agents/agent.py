
import numpy as np

class Agent:
    def __init__(self, env, gamma=0.99, alpha=0.01, eps=0):
        self.q_values = {}
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.env = env

    # state_action è la tupla (S, A)
    def get_qval(self, state_action):
        if state_action not in self.q_values:
            self.q_values[state_action] = np.random.rand(-10, 10) # ricompensa casuale, modificare valore

        return self.q_values[state_action]

