from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent
from tqdm import tqdm
from utils import *


def train(num_episodes, env, agent):
    game_scores = worker(num_episodes, env, agent, training=True)
    return game_scores


def test(num_episodes, env, agent):
    game_scores = worker(num_episodes, env, agent, training=False)
    return game_scores


def worker(num_episodes, env, agent, training):
    game_scores = []
    with tqdm(total=num_episodes, desc="Training Episodes") as progress_bar:
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
    
    game_scores_dict = {}
    agents_dict = {}

    for params in param_combinations:
        if algo == 'sarsa':
            agent = SARSAAgent(env, params['gamma'], params['alpha'], params['epsilon'])
        elif algo == 'sarsaL':
            agent = SARSALAgent(env, params['gamma'], params['alpha'], params['epsilon'], params['lambda'])
        elif algo == 'ql':
            agent = QLAgent(env, params['gamma'], params['alpha'], params['epsilon'])
        else:
            return

        game_scores = train(num_episodes, env, agent)
        game_scores_dict[tuple(params.items())] = game_scores
        agents_dict[tuple(params.items())] = agent

    plot_score(game_scores_dict, f"{algo} Training Performance", plot_filename)

    # Find the best parameter configuration
    best_params = max(
        game_scores_dict.items(),
        key=lambda item: sum(item[1]) / len(item[1])  # Compute average score
    )[0]

    # Convert the tuple of parameters back to a dictionary
    best_params_dict = dict(best_params)

    # Return the agent with the best parameters
    best_agent = agents_dict[best_params]

    return best_agent, best_params_dict


# lam è utilizzato solo se algo == 'sarsaL'
def main(algo, gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
    env = get_init_env()

    param_combinations_ql_sarsa = [
        {"gamma": 0.1, "alpha": 0.1, "epsilon": 0.7},
        {"gamma": 0.5, "alpha": 0.5, "epsilon": 0.8},
        {"gamma": 0.9, "alpha": 0.9, "epsilon": 0.9},
    ]

    param_combinations_sarsaL = [
        {"gamma": 0.1, "alpha": 0.1, "epsilon": 0.7, "lambda": 0.2},
        {"gamma": 0.5, "alpha": 0.5, "epsilon": 0.8, "lambda": 0.5},
        {"gamma": 0.9, "alpha": 0.9, "epsilon": 0.9, "lambda": 0.9},
    ]
    
    episodes = 3

    if algo == 'sarsa':
        agent, params = train_and_plot(algo, param_combinations_ql_sarsa, env, episodes, f"pretrained_models/model_{algo}/plot_train.png")
    elif algo == 'sarsaL':
        agent, params = train_and_plot(algo, param_combinations_sarsaL, env, episodes, f"pretrained_models/model_{algo}/plot_train.png")
    elif algo == 'ql':
        agent, params = train_and_plot(algo, param_combinations_ql_sarsa, env, episodes, f"pretrained_models/model_{algo}/plot_train.png")
    else:
        return
    
    print(agent.gamma)
    print("Optimal Parameters:", params)
    
    game_scores_test = test(1, env, agent)

    plot_score_test(game_scores_test, "Testing", f"pretrained_models/model_{algo}/plot_test.png")
    # plot_value_function(agent, f"pretrained_models/model_{algo}/plot_valuefunction.png")
    
    agent.save_model(agent.q_values, f"pretrained_models/model_{algo}/q_values.json")

    agent.q_values = agent.load_model(f"pretrained_models/model_{algo}/q_values.json")
    
    env.close()

if __name__ == "__main__":
    main("sarsa")


