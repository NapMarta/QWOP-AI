import sys
import os
import statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent
from agents.MC_agent import MCAgent
import MC_training
import SARSA_training
from utils import *
from threadAlgo import *
from threadCombination import *


def get_best_combination_with_scores(game_scores_dict):
    last_scores_mean = dict()
    for combination, scores in game_scores_dict.items():
        last_scores_mean[combination] = statistics.mean(scores[-10:])

    # Restituisce la combinazione che ha consentito di ottenere la massima media (calcolata sugli ultimi 10 score
    # di ogni combinazione), e tutti gli score associati a quella combinazione
    best_combination = max(last_scores_mean, key=last_scores_mean.get)
    return best_combination, game_scores_dict[best_combination]


def create_agent_by_combination(algo, env, combination):
    if algo == 'sarsa':
        agent = SARSAAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
    elif algo == 'sarsaL':
        agent = SARSALAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'], combination['lambda'])
    elif algo == 'ql':
        agent = QLAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
    elif algo == 'mc':
        agent = MCAgent(env, combination['gamma'], combination['epsilon'])
    else:
        agent = None

    return agent


def worker(algo, gamma, alpha, eps, lam, end_step):
    if algo == 'mc':
        train = MC_training.train
        test = MC_training.test
    else:
        train = SARSA_training.train
        test = SARSA_training.test

    env = get_init_env()
    num_training_episodes, num_testing_episodes = 2, 2   # 1000, 100

    game_scores_dict, agents_dict = {}, {}
    hyperparams = get_hyperparams(algo)
    threads_comb = []

    # for i_comb, combination in enumerate(hyperparams, start=1):
    #     thread_comb = ThreadCombination(algo, combination, train, num_training_episodes, env, end_step, i_comb, game_scores_dict, agents_dict)
    #     thread_comb.start()
    #     threads_comb.append(thread_comb)

    # # Il metodo join() attende il completamento di tutti i thread
    # for thread_comb in threads_comb:
    #     thread_comb.join()

    for i_comb, combination in enumerate(hyperparams, start=1):
        agent_for_training = create_agent_by_combination(algo, env, combination)

        # Lista dei guadagni per ogni episodio
        game_scores = train(num_training_episodes, env, agent_for_training, end_step, algo)
        
        if algo == 'mc':
            agent_for_training.save_model(f"pretrained_models/model_mc/{end_step}step/q_values_comb-{i_comb}.json", f"pretrained_models/model_mc/{end_step}step/policy_table_comb-{i_comb}.json")
        else: 
            agent_for_training.save_model(f"pretrained_models/model_{algo}/{end_step}step/q_values_comb-{i_comb}.json")

        # Dizionario le cui entry sono (k, v), con k = lista dei valori degli iperparametri, v = lista di score nei vari
        # episodi di training
        game_scores_dict[tuple(combination.items())] = game_scores

        # Dizionario degli agenti creati per ogni combinazione di parametri
        agents_dict[tuple(combination.items())] = agent_for_training


    # Plot dei risultati del training
    plot_score(game_scores_dict, f"{get_algo_str(algo)} Training Performance with {end_step} step", f"results/model_{algo}/{end_step}step/plot_train.png")

    best_combination, best_combination_scores_training = get_best_combination_with_scores(game_scores_dict)
    best_combination = dict(best_combination)
    print(f"Best combination for {algo} is {best_combination}")
    agent_for_testing = create_agent_by_combination(algo, env, best_combination)
    game_scores_testing = test(num_testing_episodes, env, agent_for_testing, algo)
    game_scores_testing_dict = {tuple(best_combination.items()): game_scores_testing}
    plot_score(game_scores_testing_dict, f"{get_algo_str(algo)} Testing Performance", f"results/model_{algo}/{end_step}step/plot_test.png")

    # Utilizzare per effettuare fase di training/testing di un agente pretrained
    # if algo == 'mc':
    #     agent_for_training.load_model("pretrained_models/model_MC/q_values.json", "pretrained_models/model_MC/policy_table.json")
    # else:
    #     agent_for_training.load_model(f"pretrained_models/model_{algo}/q_values.json")
    
    env.close()

    # Restituisce la tripla (migliore combinazione, lista dei guadagni per la migliore combinazione in fase di training,
    # lista dei guadagni per la migliore combinazione in fase di testing)
    return best_combination, best_combination_scores_training, game_scores_testing


# lam è utilizzato solo se algo == 'sarsaL'
def main(gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
    # algos = ['mc', 'sarsa']
    algos = ['mc', 'sarsa', 'sarsaL', 'ql']
    end_step = 4000     # 4000, 6000, 8000

    # Dizionario di coppie (k, v), con k = (algoritmo, best_combination), v = [best_combination_scores_training, game_scores_testing]
    best_by_algo = dict()

    # Dizionario per salvare i risultati condivisi
    results = {}
    threads_algo = []

    # Creazione e avvio dei thread
    for algo in algos:
        thread_algo = ThreadAlgo(algo, gamma, alpha, eps, lam, end_step, results)
        thread_algo.start()
        threads_algo.append(thread_algo)

    # Il metodo join() attende il completamento di tutti i thread
    for thread_algo in threads_algo:
        thread_algo.join()

    for algo in results:
        best_by_algo = {
            (algo, tuple(results[algo][0].items())): [results[algo][1], results[algo][2]]  
        }

    # for algo in algos:
    #     print(f"\n\n#### Execute {algo} ####")
    #     tmp_res = worker(algo, gamma, alpha, eps, lam, end_step)
    #     best_by_algo[(algo, tuple(tmp_res[0].items()))] = [tmp_res[1], tmp_res[2]]

    best_by_algo_training = {k: v[0] for k, v in best_by_algo.items()}
    best_by_algo_testing = {k: v[1] for k, v in best_by_algo.items()}

    plot_score_all_algos(best_by_algo_training, f'Training Performance with {end_step} step', f"results/all_models/{end_step}step/plot_train.png")
    plot_score_all_algos(best_by_algo_testing, 'Testing Performance', f"results/all_models/{end_step}step/plot_test.png")


if __name__ == '__main__':
    main()

