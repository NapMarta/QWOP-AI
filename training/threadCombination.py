from threading import Thread
import main_training 
from threading import Lock


class ThreadCombination(Thread):
    def __init__(self, algo, combination, train_func, num_training_episodes, env, end_step, i_comb, game_scores_dict, agents_dict):
        super().__init__()
        self.algo = algo
        self.combination = combination
        self.train_func = train_func
        self.num_training_episodes = num_training_episodes
        self.env = env
        self.end_step = end_step
        self.i_comb = i_comb
        self.game_scores_dict = game_scores_dict
        self.agents_dict = agents_dict

    def run(self):
        lock = Lock()

        agent_for_training = main_training.create_agent_by_combination(self.algo, self.env, self.combination)
        
        # Lista dei guadagni per ogni episodio
        game_scores = self.train_func(self.num_training_episodes, self.env, agent_for_training, self.end_step, self.algo)

        if self.algo == 'mc':
            agent_for_training.save_model(
                f"pretrained_models/model_mc/{self.end_step}step/q_values_comb-{self.i_comb}.json",
                f"pretrained_models/model_mc/{self.end_step}step/policy_table_comb-{self.i_comb}.json"
            )
        else:
            agent_for_training.save_model(f"pretrained_models/model_{self.algo}/{self.end_step}step/q_values_comb-{self.i_comb}.json")

        with lock:
            # Aggiornamento dei dizionari condivisi

            # Dizionario le cui entry sono (k, v), con k = lista dei valori degli iperparametri, v = lista di score nei vari
            # episodi di training
            self.game_scores_dict[tuple(self.combination.items())] = game_scores
            
            # Dizionario degli agenti creati per ogni combinazione di parametri
            self.agents_dict[tuple(self.combination.items())] = agent_for_training


