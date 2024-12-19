from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent
from tqdm import tqdm
from .utils import *


def train(num_episodes, env, agent):
    return worker(num_episodes, env, agent, training=True)


def test(num_episodes, env, agent):
    return worker(num_episodes, env, agent, training=False)


def worker(num_episodes, env, agent, training):
    game_scores = []
    desc = "Training episodes:" if training else "Testing episode:"
    with tqdm(total=num_episodes, desc=desc) as progress_bar:
        for i in range(num_episodes):
            episode_reward = 0
            curr_state = agent.export_state(agent.env.reset()[0])
            curr_action = agent.get_action(curr_state, training)

            while True:
                next_state, reward, terminated, truncated, info = env.step(curr_action)
                next_state = agent.export_state(next_state)
                next_action = agent.get_action(next_state, training)

                # Reward personalizzata
                episode_reward += agent.reward_function(reward, info)
                # print(episode_reward)

                if training:
                    # A seconda che l'agent sia di tipo SARSA, SARSA-L, Q-Learing, viene eseguita
                    # la specifica funzione di aggiornamento, implementata nello specifico agente
                    agent.update_qval(curr_state, curr_action, reward, terminated or truncated, next_state, next_action)

                if terminated or truncated:
                    break

                curr_state, curr_action = next_state, next_action

            game_scores.append(episode_reward)
            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
    
    return game_scores


def train_and_plot(algo, param_combinations, env, num_episodes, plot_filename):
    pass
    #
    #
    # # Find the best parameter configuration
    # best_params = max(
    #     game_scores_dict.items(),
    #     key=lambda item: sum(item[1]) / len(item[1])  # Compute average score
    # )[0]
    #
    # # Convert the tuple of parameters back to a dictionary
    # best_params_dict = dict(best_params)
    #
    # # Return the agent with the best parameters
    # best_agent = agents_dict[best_params]
    #
    # return best_agent, best_params_dict


def get_best_combination(game_scores_dict):
    last_scores = dict()
    for combination, scores in game_scores_dict.items():
        last_scores[combination] = scores[len(scores) - 1]

    # Restituisce la combinazione che ha consentito di ottenere il massimo last score
    return max(last_scores, key=last_scores.get)


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


# lam è utilizzato solo se algo == 'sarsaL'
def main(algo, gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
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

    best_combination = get_best_combination(game_scores_dict)
    agent_for_testing = create_agent_by_combination(algo, env, best_combination)
    game_scores = test(num_training_episodes, env, agent_for_testing)
    game_scores_dict[tuple(combination.items())] = game_scores
    plot_score_test(game_scores_dict, f"{algo} Testing performance", f"pretrained_models/model_{algo}/plot_test.png")



    # print(agent.gamma)
    # print("Optimal Parameters:", params)

    # game_scores_test = test(num_testing_episodes, env, agent)
    #
    # plot_score_test(game_scores_test, "Testing", f"pretrained_models/model_{algo}/plot_test.png")
    # agent.save_model(agent.q_values, f"pretrained_models/model_{algo}/q_values.json")
    #
    #
    # game_scores_train = train(5, env, agent)
    # plot_score(game_scores_train, "Training", f"pretrained_models/model_{algo}/plot_train.png")
    #
    # game_scores_test = test(2, env, agent)
    #
    # plot_score_test(game_scores_test, "Testing", f"pretrained_models/model_{algo}/plot_test.png")
    # # plot_value_function(agent, f"pretrained_models/model_{algo}/plot_valuefunction.png")
    #
    # agent.save_model(agent.q_values, f"pretrained_models/model_{algo}/q_values.json")
    #
    # agent.q_values = agent.load_model(f"pretrained_models/model_{algo}/q_values.json")
    #
    env.close()



if __name__ == "__main__":
    main("sarsa")


