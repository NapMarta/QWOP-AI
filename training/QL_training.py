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
            curr_action = agent.get_action(curr_state, training)

        print(f'Terminato episodio {i+1} con ricompensa {episode_reward}')


def testing(env, agent, test_episodes):
    return 


if __name__ == "__main__":
    #Scegliere valori ottimali
    gamma=0.1
    alpha=0.1
    eps=0.2
    episodes = 10

    env = utils.get_init_env()
    
    # Create a new agent
    agentQL = QLAgent(env, gamma, alpha, eps)    
    print("Training:")
    training(env, agentQL, episodes)

    # print("Testing:")
    # testing(env, agentQL, episodes)
    env.close()


