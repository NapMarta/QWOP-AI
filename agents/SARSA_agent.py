
from agents.agent import Agent

class SARSAAgent(Agent):
    def __init__(self, env, gamma, alpha, eps):
        super().__init__(env, gamma, eps)
        self.alpha = alpha


    def update_qval(self, curr_state, curr_action, reward, terminal, next_state=None, next_action=None):
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.get_qval((next_state, next_action))

        td_error = td_target - self.get_qval((curr_state, curr_action))
        self.q_values[(curr_state, curr_action)] += self.alpha * td_error


