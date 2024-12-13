import numpy as np
from agents.agent import Agent

class QLAgent(Agent):
    def __init__(self, env, gamma, alpha, eps):
        super().__init__(env, gamma, eps)
        self.alpha = alpha


    # Funzione che effettua l'aggiornamento della action-value function nel Q-Learning
    # unused_next_action è inutilizzato: serve a realizzare il polimorfismo
    def update_qval(self, curr_state, action, reward, terminal, next_state, unused_next_action):
        if terminal:
            td_target = reward
        else:
            next_action = self.get_action(next_state, False)
            max_q_next = self.get_qval((next_state, next_action))
            td_target = reward + self.gamma * max_q_next

        td_error = td_target - self.get_qval((curr_state, action))
        self.q_values[(curr_state, action)] += self.alpha * td_error

