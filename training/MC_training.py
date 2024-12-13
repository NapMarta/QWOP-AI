from agents.MC_agent import MCAgent
import numpy as np
from tqdm import tqdm
import qwop_gym
import gymnasium as gym
from collections import defaultdict
import utils

def train(env, agent, num_episodes):
    
    # Inizializzazione della policy
    # policy = make_epsilon_greedy_policy(agent)

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    episode_reward = 0
    
    with tqdm(total=num_episodes, desc="Training Episodes") as progress_bar:
        for _ in range(num_episodes):
            episodes = []
            # Genera gli episodi fino a raggiungere uno stato finale 
            # An episode is an array of (statec, action, reward) tuples
            state_idx = agent.export_state(agent.env.reset()[0])
            # for i in range(500):
            while True:
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

            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})


def test(env, agent, num_episodes):
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

            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
            


if __name__ == "__main__":
    env = utils.get_init_env()
    
    # Create a new agent
    agentMC = MCAgent(env = env)    
    print("Training: ")
    train(env, agentMC, 1)

    print("Testing:")
    test(env, agentMC, 1)
    env.close()

