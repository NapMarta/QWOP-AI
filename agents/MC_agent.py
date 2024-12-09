
import numpy as np
from agents import Agent
import qwop_gym
import gymnasium as gym
from collections import defaultdict

class MCAgent(Agent):
    def __init__(self, env, gamma=1.0, alpha=0, eps=0.1):
        # il fattore alpha non serve
        super().__init__(env, gamma, alpha, eps)
        self.n_actions = 9
        self.policy_table = defaultdict(lambda: np.ones(self.n_actions) * (self.eps / self.n_actions))


    def get_action(self, curr_state):
        probs = self.policy_table[curr_state]
        probs /= np.sum(probs)  # Normalizza prima di usarle
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action


    # Funzione per aggiornare le probabilità della policy ϵ-greedy
    def update_policy(self, state):
        # Aggiorna la policy per essere ϵ-greedy rispetto a Q(s, a)
        q_values = [self.get_qval((state, a)) for a in range(self.n_actions)]
        best_action = np.argmax(q_values)
        for a in range(self.n_actions):
            if a == best_action:
                self.policy_table[state][a] = 1 - self.eps + (self.eps / self.n_actions)
            else:
                self.policy_table[state][a] = self.eps / self.n_actions

        # Normalizza le probabilità per evitare errori
        self.policy_table[state] /= np.sum(self.policy_table[state])





