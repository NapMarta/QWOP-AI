
import numpy as np

class Agent:
    def __init__(self, env, gamma, alpha, eps):
        self.q_values = dict()
        self.states_vect = []
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.env = env

    # state_action è la tupla (S, A)
    def get_qval(self, state_action):
        if state_action not in self.q_values:
            self.q_values[state_action] = np.random.uniform(-10, 10) # ricompensa casuale, modificare valore
        return self.q_values[state_action]

    def export_state(self, state):
        st_index = -1
        for i in range(len(self.states_vect)):
            if np.array_equal(state, self.states_vect[i]):
                st_index = i
                break
        if st_index == -1:
            st_index = len(self.states_vect)
            self.states_vect.append(state)
        return st_index

    def get_action(self, curr_state):
        if np.random.rand() < self.eps:
            # Fai esplorazione
            # print('esplorazione:')
            next_action = self.env.action_space.sample()
        else:
            # Fai sfruttamento
            # print('sfruttamento:', self.eps)
            # Restituisce l'indice dell'azione a cui è associato il massimo q_value
            q_values = []
            for i in range(9):
                q_values.append(self.get_qval((curr_state, i)))
            next_action = np.argmax(q_values)

        # Decrementa leggermente la prob. di esplorazione
        self.eps *= 0.99
        return next_action


