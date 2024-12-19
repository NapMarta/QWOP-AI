from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent
from tqdm import tqdm
from .utils import *


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


# lam è utilizzato solo se algo == 'sarsaL'
def main(algo, gamma=0.1, alpha=0.1, eps=0.2, lam=0.2):
    env = get_init_env()

    if algo == 'sarsa':
        agent = SARSAAgent(env, gamma, alpha, eps)
    elif algo == 'sarsaL':
        agent = SARSALAgent(env, gamma, alpha, eps, lam)
    elif algo == 'ql':
        agent = QLAgent(env, gamma, alpha, eps)
    else:
        return

    game_scores_train = train(5, env, agent)
    plot_score(game_scores_train, "Training", f"pretrained_models/model_{algo}/plot_train.png")

    game_scores_test = test(1, env, agent)

    plot_score(game_scores_test, "Testing", f"pretrained_models/model_{algo}/plot_test.png")
    plot_value_function(agent, f"pretrained_models/model_{algo}/plot_valuefunction.png")

    agent.save_model(f"pretrained_models/model_{algo}/q_values.json")

    agent.q_values = agent.load_model(f"pretrained_models/model_{algo}/q_values.json")
    
    env.close()



