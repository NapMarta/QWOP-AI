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

def train(num_episodes, env, agent, end_step, algo):
    
    # Inizializzazione della policy
    # policy = make_epsilon_greedy_policy(agent)

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    episode_reward = 0
    game_scores = []
    avgspeed = []
    
    with tqdm(total=num_episodes, desc="Training Episodes " + algo) as progress_bar:
        for _ in range(num_episodes):
            speed = 0
            episodes = []
            # Genera gli episodi fino a raggiungere uno stato finale 
            # An episode is an array of (statec, action, reward) tuples
            state_idx = agent.export_state(agent.env.reset()[0])
            for i in range(end_step):
            # while True:
                action = agent.get_action(state_idx, exploration = True)
                next_state, reward_env, terminated, truncated, info = env.step(action)
                speed += info['avgspeed']
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
            avgspeed.append(speed/end_step)
            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
    
    return game_scores, avgspeed



def test(num_episodes, env, agent, algo):
    game_scores = []
    avgspeed = []

    with tqdm(total=num_episodes, desc="Testing Episodes " + algo) as progress_bar:
        for i in range(num_episodes):
            episode_reward = 0
            speed = 0 
            curr_state = agent.export_state(agent.env.reset()[0])
            curr_action = agent.get_action(curr_state)
            step = 0

            while True:
                next_state, reward, terminated, truncated, info = env.step(curr_action)
                speed += info['avgspeed']
                next_state = agent.export_state(next_state)
                next_action = agent.get_action(next_state)
                step += 1

                # Reward personalizzata
                episode_reward += agent.reward_function(reward, info)
                # print(episode_reward)

                if terminated or truncated:
                    break

                curr_state, curr_action = next_state, next_action

            game_scores.append(episode_reward)
            avgspeed.append(speed/step)
            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
        
    
    return game_scores, avgspeed


if __name__ == "__main__":
    env = get_init_env()
    combination = get_hyperparams("mc")[0]
    agent = MCAgent(env, combination['gamma'], combination['epsilon'])
    agent.load_model("pretrained_models/model_mc/3000step/q_values_comb-3.json", "pretrained_models/model_mc/3000step/policy_table_comb-3.json" )

    game_scores, avgspeed = test(5, env, agent, "mc")

    print(f"game scores: {game_scores} \navg speed: {avgspeed}")
    env.close()