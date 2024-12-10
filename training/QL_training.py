import utils 
from agents.QL_agent import QLAgent
import numpy as np
from utils import *


def training(env, agent, episodes):
    for i in range(episodes):
        episode_reward = 0
        curr_state = agent.export_state(agent.env.reset()[0])
        curr_action = agent.get_action(curr_state, True)

        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step(curr_action)
            next_state = agent.export_state(next_state)
            
            reward_update = agent.reward_function(reward, info)
            episode_reward += reward_update

            agent.td_update(curr_state, curr_action, reward_update, next_state)

            if terminated or truncated:
                    break

            curr_state = next_state
            curr_action = agent.get_action(curr_state, True)

        print(f'Terminato episodio {i+1} con ricompensa {episode_reward}')


def testing(env, agent, test_episodes):
    total_rewards = []

    for i in range(test_episodes):
        episode_reward = 0
        curr_state = agent.export_state(env.reset()[0])
        curr_action = agent.get_action(curr_state, False) 
        
        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step(curr_action)
            next_state = agent.export_state(next_state)
            
            reward_update = agent.reward_function(reward, info)
            episode_reward += reward_update

            if terminated or truncated:
                break

            curr_state = next_state
            curr_action = agent.get_action(curr_state, False)

        total_rewards.append(episode_reward)
        
        print(f'Terminato episodio di test {i+1} con ricompensa {episode_reward}')

    average_reward = np.mean(total_rewards)
    print(f'Reward media su {test_episodes} episodi di test: {average_reward}')

    return total_rewards, average_reward

if __name__ == "__main__":
    # Valore iniziale di gamma vicino a 1, bilancia bene ricompense immediate e future
    gamma = 0.95
    alpha = 0.2
    # Valore iniziale ad 1 con lieve decadimento ogni volta che facciamo esplorazione
    eps = 1.0
    episodes = 10

    env = utils.get_init_env()
    
    # Create a new agent
    agentQL = QLAgent(env, gamma, alpha, eps)    
    print("Training con Q-Learning:")
    training(env, agentQL, episodes)

    print("Testing con Q-Learning:")
    testing(env, agentQL, episodes)
    env.close()


