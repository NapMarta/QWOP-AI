
import json
import numpy as np
from agents.agent import Agent
import qwop_gym
import gymnasium as gym
from collections import defaultdict
import os

class MCAgent(Agent):
    def __init__(self, env, gamma=1.0, eps=0.1):
        # il fattore alpha non serve
        super().__init__(env, gamma, eps)
        self.n_actions = 9
        # key - lista di probabilità per azioni
        self.policy_table = defaultdict(lambda: np.ones(self.n_actions) * (self.eps / self.n_actions))


    def get_action(self, curr_state, exploration = False):
        probs = self.policy_table[curr_state]
        probs /= np.sum(probs)  # Normalizzazione

        if exploration:
            action = np.random.choice(np.arange(len(probs)), p=probs)
        else:
            action = np.argmax(probs)
            
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


    def save_model(self, filename_qvalues, filename_policy):
        q_table_str_keys = {str(key): value for key, value in self.q_values.items()}

        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filename_qvalues), exist_ok=True)
        with open(filename_qvalues, "w") as file:
            json.dump(q_table_str_keys, file)


        policy_keys = {str(key): value.tolist() for key, value in self.policy_table.items()}

        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filename_policy), exist_ok=True)
        with open(filename_policy, "w") as file:
            # print(policy_keys)
            json.dump(policy_keys, file)


    def load_model(self, filename_qvalues, filename_policy):
        with open(filename_qvalues, "r") as file:
            q_table_str_keys = json.load(file)

        self.q_values = {eval(key): value for key, value in q_table_str_keys.items()}
        
        with open(filename_policy, "r") as file:
            tmp_policy = json.load(file)

        # in questo modo se lo stato non era presente, usiamo la probabilità di default
        for key, value in tmp_policy.items():
            self.policy_table[str(key)] = value

