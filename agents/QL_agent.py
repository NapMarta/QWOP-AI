import numpy as np
from agents.agent import Agent

class QLAgent(Agent):
    def __init__(self, env, gamma, alpha, eps):
        super().__init__(env, gamma, alpha, eps)


    # Funzione che effettua l'aggiornamento della 
    # action-value function nel Q-Learning
    def td_update(self, curr_state, action, reward, next_state):
        
        # actions = set(action for (_, action) in self.q_values.keys())
        
        # if not actions:
        #     max_q_next = 0
        # else:
        #     # Prende il massimo tra tutte le possibili azioni eseguibili dallo stato s'
        #     max_q_next = max([self.q_values.get((next_state, action), 0) for action in actions])
        #     print(max_q_next.value())

        next_action = self.get_action(next_state, True)

        if not next_action:
            max_q_next =  0
        else:   
            max_q_next = self.get_qval((next_state, next_action))
        
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.get_qval((curr_state, action))
        self.q_values[(curr_state, action)] += self.alpha * td_error

