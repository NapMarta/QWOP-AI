from agents.SARSAL_agent import SARSALAgent
from .utils import *


def train(num_episodes, env, agent):
    for i in range(num_episodes):
        episode_reward = 0
        curr_state = agent.export_state(agent.env.reset()[0])
        curr_action = agent.get_action(curr_state)

        while True:
            next_state, reward, terminated, truncated, info = env.step(curr_action)
            next_state = agent.export_state(next_state)
            next_action = agent.get_action(next_state)

            # Reward personalizzata
            episode_reward += reward + info['distance']/100
            # print(episode_reward)

            agent.update_all(curr_state, curr_action, reward, terminated, next_state, next_action)

            if terminated or truncated:
                break

            curr_state, curr_action = next_state, next_action

        print(f'Terminato episodio {i+1} con ricompensa {episode_reward}')


def main(gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
    env = get_init_env()
    agent = SARSALAgent(env, gamma, alpha, eps, lam)
    train(10, env, agent)
    env.close()



