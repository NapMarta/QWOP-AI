import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.MC_agent import MCAgent
import numpy as np
from tqdm import tqdm
import qwop_gym
import gymnasium as gym
from collections import defaultdict
from utils import *

def train(env, agent, num_episodes):
    
    # Inizializzazione della policy
    # policy = make_epsilon_greedy_policy(agent)

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    episode_reward = 0
    game_scores = []
    
    with tqdm(total=num_episodes, desc="Training Episodes") as progress_bar:
        for _ in range(num_episodes):
            episodes = []
            # Genera gli episodi fino a raggiungere uno stato finale 
            # An episode is an array of (statec, action, reward) tuples
            state_idx = agent.export_state(agent.env.reset()[0])
            for i in range(1000):
            # while True:
                action = agent.get_action(state_idx, exploration = True)
                next_state, reward_env, terminated, truncated, info = env.step(action)
                reward = agent.reward_function(reward_env, info)
                episodes.append((state_idx, action, reward))
                if terminated or truncated:
                    break
                state_idx = agent.export_state(next_state)

            visited_state_actions = set()  # Per tracciare la prima visita


            # Itero dall'ultimo step dell'episodio
            for t, (state, action, reward) in enumerate(reversed(episodes)): 
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    # Calcola il ritorno G a partire dallo stato corrente t
                    G = sum([x[2] * (agent.gamma ** i) for i, x in enumerate(episodes[t:])])
                    

                    # Aggiorna le somme e i contatori per calcolare la media
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1

                    # Aggiorna il valore Q(s, a) usando la media incrementale
                    agent.q_values[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]
                    if(t == len(episodes)-1):
                        episode_reward = episodes[t][2]

                    agent.update_policy(state)

            game_scores.append(episode_reward)
            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
    
    return game_scores



def test(env, agent, num_episodes):
    game_scores = []
    with tqdm(total=num_episodes, desc="Testing Episodes") as progress_bar:
        for i in range(num_episodes):
            episode_reward = 0
            curr_state = agent.export_state(agent.env.reset()[0])
            curr_action = agent.get_action(curr_state)

            while True:
                next_state, reward, terminated, truncated, info = env.step(curr_action)
                next_state = agent.export_state(next_state)
                next_action = agent.get_action(next_state)

                # Reward personalizzata
                episode_reward += agent.reward_function(reward, info)
                # print(episode_reward)

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
        agent = MCAgent(env, params['gamma'], params['epsilon'])
        
        game_scores = train(num_episodes, env, agent)
        game_scores_dict[tuple(params.items())] = game_scores
        agents_dict[tuple(params.items())] = agent

    plot_score(game_scores_dict, f"MonteCarlo Training Performance", plot_filename)

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



if __name__ == "__main__":
    env = get_init_env()

    param_combinations_mc = [
        {"gamma": 0.1, "epsilon": 0.2},
        {"gamma": 0.5, "epsilon": 0.3},
        {"gamma": 0.9, "epsilon": 0.1},
    ]
    
    # Create a new agent
    # agentMC = MCAgent(env = env)    
    print("Training: ")

    episodes = 5
    
    agentMC, params = train_and_plot(param_combinations_mc, env, episodes, "pretrained_models/model_MC/plot_train.png")

    # game_scores_train = train(env, agentMC, 1)
    # utils.plot_score(game_scores_train, "Training", "pretrained_models/model_MC/plot_train.png")

    agentMC.save_model("pretrained_models/model_MC/q_values.json", "pretrained_models/model_MC/policy_table.json")

    agentMC.load_model("pretrained_models/model_MC/q_values.json", "pretrained_models/model_MC/policy_table.json")
    
    print("Testing:")
    game_scores_test = test(env, agentMC, 2)

    plot_score_test(game_scores_test, "Testing", "pretrained_models/model_MC/plot_test.png")
    # utils.plot_value_function(agentMC, "pretrained_models/model_MC/plot_valuefunction.png")

    env.close()

