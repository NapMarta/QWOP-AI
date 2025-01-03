
import json
import numpy as np
import os

class Agent:
    def __init__(self, env, gamma, eps):
        # dizionario dove la chiave è (id_stato, azione) e il valore è l'action-value
        self.q_values = dict()
        self.states_vect = []
        self.gamma = gamma
        self.eps = eps
        self.env = env

    # Funzione di ricompensa personalizzata
    def reward_function(self, reward_env, info):
        reward = reward_env + (info['distance'] // 10) + (info['avgspeed'] // 10)
        return reward

    # state_action è la tupla (S, A)
    def get_qval(self, state_action):
        if state_action not in self.q_values:
            self.q_values[state_action] = np.random.uniform(-10, 10) # ricompensa casuale, modificare valore
        return self.q_values[state_action]

    # Mappa lo stato (matrice 12x5) in un intero univoco progressivo
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

    def get_action(self, curr_state, exploration):
        if exploration and np.random.rand() < self.eps:
            # Fai esplorazione
            next_action = self.env.action_space.sample()
            # Decrementa leggermente la prob. di esplorazione
            self.eps *= 0.99
        else:
            # Fai sfruttamento
            # Se sto facendo testing, non ho la componente di esplorazione: voglio fare solo sfruttamento
            # perché mi serve valutare la policy ottenuta
            # Restituisce l'indice dell'azione a cui è associato il massimo q_value
            q_values = []
            for i in range(9):
                q_values.append(self.get_qval((curr_state, i)))
            next_action = np.argmax(q_values)

        return next_action


    def save_model(self, filename):
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        q_table_str_keys = {str(key): value for key, value in self.q_values.items()}

        with open(filename, "w") as file:
            json.dump(q_table_str_keys, file)


    def load_model(self, filename):
        with open(filename, "r") as file:
            q_table_str_keys = json.load(file)

        self.q_values = {eval(key): value for key, value in q_table_str_keys.items()}
        

