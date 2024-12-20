from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent
from agents.MC_agent import MCAgent
from training.SARSA_training import main as SARSA_main
import MC_training
import SARSA_training
from .utils import *


def get_best_combination_with_scores(game_scores_dict):
    last_scores = dict()
    for combination, scores in game_scores_dict.items():
        last_scores[combination] = scores[len(scores) - 1]

    # Restituisce la combinazione che ha consentito di ottenere il massimo last score, e tutti gli score associati
    # a quella combinazione
    best_combination = max(last_scores, key=last_scores.get)
    return best_combination, game_scores_dict[best_combination]


def create_agent_by_combination(algo, env, combination):
    if algo == 'sarsa':
        agent = SARSAAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
    elif algo == 'sarsaL':
        agent = SARSALAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'], combination['lambda'])
    elif algo == 'ql':
        agent = QLAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
    else:
        agent = None

    return agent


def worker(algo, gamma, alpha, eps, lam):
    if algo == 'mc':
        train = MC_training.train
        test = MC_training.test
    else:
        train = SARSA_training.train
        test = SARSA_training.test

    env = get_init_env()
    num_training_episodes, num_testing_episodes = 2, 1

    game_scores_dict, agents_dict = {}, {}
    hyperparams = get_hyperparams(algo)

    for i_comb, combination in enumerate(hyperparams, start=1):
        agent_for_training = create_agent_by_combination(algo, env, combination)

        # Lista dei guadagni per ogni episodio
        game_scores = train(num_training_episodes, env, agent_for_training)
        agent_for_training.save_model(f"pretrained_models/model_{algo}/q_values_comb-{i_comb}.json")

        # Dizionario le cui entry sono (k, v), con k = lista dei valori degli iperparametri, v = lista di score nei vari
        # episodi di training
        game_scores_dict[tuple(combination.items())] = game_scores

        # Dizionario degli agenti creati per ogni combinazione di parametri
        agents_dict[tuple(combination.items())] = agent_for_training


    # Plot dei risultati del training
    plot_score(game_scores_dict, f"{algo} Training performance", f"pretrained_models/model_{algo}/plot_train.png")

    best_combination, best_combination_scores_training = get_best_combination_with_scores(game_scores_dict)
    best_combination = dict(best_combination)
    print(best_combination)
    agent_for_testing = create_agent_by_combination(algo, env, best_combination)
    game_scores_testing = test(num_training_episodes, env, agent_for_testing)
    game_scores_testing_dict = {tuple(best_combination.items()): game_scores_testing}
    plot_score(game_scores_testing_dict, f"{algo} Testing performance", f"pretrained_models/model_{algo}/plot_test.png")

    # agent.q_values = agent.load_model(f"pretrained_models/model_{algo}/q_values.json")
    env.close()

    # Restituisce la tripla (migliore combinazione, lista dei guadagni per la migliore combinazione in fase di training,
    # lista dei guadagni per la migliore combinazione in fase di testing)
    return best_combination, best_combination_scores_training, game_scores_testing


# lam è utilizzato solo se algo == 'sarsaL'
def main(gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
    algos = ['mc']
    # algos = ['sarsa', 'sarsaL', 'ql']

    # Dizionario di coppie (k, v), con k = algoritmo, v = [best_combination, best_combination_scores_training, game_scores_testing]
    best_by_algo = dict()

    for algo in algos:
        best_by_algo[algo] = worker(algo, gamma, alpha, eps, lam)

    print(best_by_algo['mc'])

if __name__ == '__main__':
    main()

