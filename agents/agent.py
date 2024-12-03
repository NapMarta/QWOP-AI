
import numpy as np

class Agent:
    def __init__(self, env, gamma, alpha, eps):
        self.q_values = {}
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.env = env

    # state_action è la tupla (S, A)
    def get_qval(self, state_action):
        if state_action not in self.q_values:
            self.q_values[state_action] = np.random.rand(0, 10) # ricompensa casuale, modificare valore

        return self.q_values[state_action]

    def get_action(self, curr_state):
        if np.random.rand() < self.eps:
            # Fai esplorazione
            next_action = self.env.action_space.sample()
        else:
            # Fai sfruttamento
            # Restituisce l'indice dell'azione a cui è associato il massimo q_value
            next_action = np.argmax([self.get_qval((curr_state, i) for i in range(9))])

        # Decrementa leggermente la prob. di esplorazione
        self.eps *= 0.99
        return next_action


