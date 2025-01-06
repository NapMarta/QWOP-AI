import sys
import os
import statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import utils
from agents.SARSA_agent import SARSAAgent
from agents.SARSAL_agent import SARSALAgent
from agents.QL_agent import QLAgent


def train(num_episodes, env, agent, end_step, algo):
    return worker(num_episodes, env, agent, algo, end_step, training=True)


def test(num_episodes, env, agent, algo):
    return worker(num_episodes, env, agent, algo, end_step = 0, training=False)


def worker(num_episodes, env, agent, algo, end_step, training):
    game_scores = []
    avgspeed = []
    desc = "Training Episodes " + algo if training else "Testing Episodes " + algo
    with tqdm(total=num_episodes, desc=desc) as progress_bar:
        for i in range(num_episodes):
            episode_reward = 0
            speed = 0
            curr_state = agent.export_state(agent.env.reset()[0])
            curr_action = agent.get_action(curr_state, training)
            step = 0

            while True:
                next_state, reward, terminated, truncated, info = env.step(curr_action)
                speed += info['avgspeed']
                next_state = agent.export_state(next_state)
                if training and step > end_step - 1:
                    break

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

                step += 1 
                curr_state, curr_action = next_state, next_action

            game_scores.append(episode_reward)
            avgspeed.append(speed/step)
            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})
    
    return game_scores, avgspeed


if __name__ == "__main__":
    algo = input("Inserisci il nome dell'algoritmo (sarsa, sarsaL, ql): ")
    
    if algo == "sarsa":
        env = utils.get_init_env()
        combination = utils.get_hyperparams("sarsa")[1]
        agent = SARSAAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
        agent.load_model("pretrained_models/model_sarsa/3000step/q_values_comb-2.json")

        game_scores, avgspeed = test(10, env, agent, "sarsa")

        print(f"game scores: {game_scores} \navg speed: {avgspeed}")
        env.close()
    
    elif algo == "sarsaL":
        env = utils.get_init_env()
        combination = utils.get_hyperparams("sarsaL")[1]
        agent = SARSALAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'],  combination['lambda'])
        agent.load_model("pretrained_models/model_sarsaL/3000step/q_values_comb-2.json")
        print(agent.q_values)
        game_scores, avgspeed = test(10, env, agent, "sarsaL")

        print(f"game scores: {game_scores} \navg speed: {avgspeed}")
        env.close()

    elif algo == "ql":
        env = utils.get_init_env()
        combination = utils.get_hyperparams("ql")[1]
        agent = QLAgent(env, combination['gamma'], combination['alpha'], combination['epsilon'])
        agent.load_model("pretrained_models/model_ql/3000step/q_values_comb-2.json")

        game_scores, avgspeed = test(10, env, agent, "ql")

        print(f"game scores: {game_scores} \navg speed: {avgspeed}")
        env.close()
