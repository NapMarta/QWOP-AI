import numpy as np
from agents.agent import Agent

class QLAgent(Agent):
    def __init__(self, env, gamma, alpha, eps):
        super().__init__(env, gamma, alpha, eps)

    # Funzione che effettua l'aggiornamento della 
    # action-value function nel Q-Learning
    def td_update(self, curr_state, action, reward, next_state):
        
        actions = set(action for (_, action) in self.q_values.keys())
        
        max_q_next = 0
        # Prende il massimo tra tutte le possibili azioni eseguibili dallo stato s'
        max_q_next = max([self.q_values.get((next_state, action), 0) for action in actions])

        
        """ if (next_state, ) in self.q_table:
            max = max(self.q_values[(next_state, )]) """
            
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.get_qval((curr_state, action))
        self.q_values[(curr_state, action)] += self.alpha * td_error

