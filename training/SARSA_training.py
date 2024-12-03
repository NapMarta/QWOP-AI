import qwop_gym
import gymnasium as gym
from agents.SARSA_agent import SARSAAgent


def get_init_env():
    env = gym.make(
        "QWOP-v1",
        browser="C:/Program Files/Google/Chrome/Application/chrome.exe",
        driver="C:/Users/fgfoo/OneDrive/Documents/Uni/magistrale/anno-2/IA/chromedriver-win64/chromedriver-win64/chromedriver.exe",
        auto_draw=True,
        stat_in_browser=True,
        reduced_action_set=True
    )

    return env


def train_forward(num_episodes, env, agent):
    for i in range(num_episodes):
        episode_reward = 0
        curr_state = agent.export_state(agent.env.reset()[0])
        curr_action = agent.get_action(curr_state)

        while True:
            next_state, reward, terminated, truncated, info = env.step(curr_action)
            next_state = agent.export_state(next_state)
            next_action = agent.get_action(next_state)
            episode_reward += reward

            agent.update_qval(curr_state, curr_action, reward, terminated, next_state, next_action)

            if terminated or truncated:
                break

            curr_state, curr_action = next_state, next_action

        print(f'Terminato episodio {i+1} con ricompensa {episode_reward}')


def main(gamma=0.1, alpha=0.1, eps=0.2):
    env = get_init_env()
    agent = SARSAAgent(env, gamma, alpha, eps)
    train_forward(10, env, agent)



