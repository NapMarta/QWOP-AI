
from agents.agent import Agent

class SARSALAgent(Agent):
    def __init__(self, env, gamma, alpha, eps, lam):
        super().__init__(env, gamma, alpha, eps)
        self.lam = lam
        self.et_values = dict()


    # state_action è la tupla (S, A)
    def get_et_val(self, state_action):
        if state_action not in self.et_values:
            self.et_values[state_action] = 0
        return self.et_values[state_action]


    def decrease_et_val(self, state_action):
        self.et_values[state_action] *= self.lam * self.gamma


    def increase_et_val(self, state_action):
        if state_action not in self.et_values:
            self.et_values[state_action] = 0

        self.et_values[state_action] += 1


    def update_all(self, curr_state, curr_action, reward, terminal, next_state=None, next_action=None):
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.get_qval((next_state, next_action))

        td_error = td_target - self.get_qval((curr_state, curr_action))
        self.increase_et_val((curr_state, curr_action))

        # Propagazione degli aggiornamenti
        for i_state in range(len(self.states_vect)):
            for j_action in range(9):
                # Il get serve ad inizializzare Q(i_state, j_action) per evitare KeyError; il risultato non è significativo
                self.get_qval((i_state, j_action))
                self.q_values[(i_state, j_action)] += self.alpha * td_error * self.get_et_val((i_state, j_action))
                self.decrease_et_val((i_state, j_action))



