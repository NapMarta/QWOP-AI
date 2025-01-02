from collections import defaultdict
import json
import matplotlib
import qwop_gym
import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_init_env(speed_rew_mult=None, time_cost__mult=None, frames_per_step=None, success_reward=None, failure_cost=None):
    browser_path, driver_path = _get_path()

    # Dizionario dei parametri da passare a gym.make
    gym_kwargs = {
        "browser": browser_path,
        "driver": driver_path,
        "auto_draw": True,
        "stat_in_browser": True,
        "reduced_action_set": True
    }

    if speed_rew_mult is not None:
        gym_kwargs["speed_rew_mult"] = speed_rew_mult
    if time_cost__mult is not None:
        gym_kwargs["time_cost__mult"] = time_cost__mult
    if frames_per_step is not None:
        gym_kwargs["frames_per_step"] = frames_per_step
    if success_reward is not None:
        gym_kwargs["success_reward"] = success_reward
    if failure_cost is not None:
        gym_kwargs["failure_cost"] = failure_cost

    env = gym.make("QWOP-v1", **gym_kwargs)
    return env


def _get_path():
    file_path = "config_env.txt"
    browser_path = ""
    driver_path = ""

    try:
        with open(file_path, 'r') as file:
            row = file.readlines()
            browser_path = row[1].strip()  
            driver_path = row[3].strip()  

    except FileNotFoundError:
        print(f"Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(e)

    print("Browser Path:", browser_path)
    print("Driver Path:", driver_path)

    return browser_path, driver_path


def get_actionValue(Q):
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value

    return V

def get_algo_str(algo):
    if algo == 'mc':
        return 'Monte-Carlo'
    elif algo == 'sarsaL':
        return 'SARSA(λ)'
    elif algo == 'sarsa':
        return 'SARSA'
    elif algo == 'ql':
        return 'Q-Learning'


def plot_score(game_scores, title, filename):
    plt.figure(figsize=(12, 8))

    for params_tuple, scores in game_scores.items():
        # Numero di episodi
        num_episodes = len(scores)
        step = max(1, num_episodes // 10)  # Calcolo del passo per sottocampionamento

        # Sottocampionamento: Seleziona 10 punti distribuiti uniformemente
        x_values = range(1, num_episodes + 1, step)
        y_values = [scores[i - 1] for i in x_values]
        
        # Label per i parametri
        params = dict(params_tuple)
        label = (
            f"γ={params['gamma']} "
            f"ε={params['epsilon']} "
        )
        if 'alpha' in params:
            label += f"α={params['alpha']} "
        if 'lambda' in params:
            label += f", λ={params['lambda']} "

        # Traccia i dati sottocampionati
        plt.plot(x_values, y_values, label=label)

    # Configura i tick sull'asse x per mostrare solo 10 valori
    plt.xticks(x_values)

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend(title="Hyperparameters", loc="best")
    plt.savefig(filename)


def plot_score_all_algos(best_by_algo, title, filename):
    # best_by_algo è un dict le cui entry sono (k, v), con k = (algoritmo, best_combination), v = [best_combination_scores]
    # Gli scores sono di training o di testing a seconda dell'invocazione da parte di main_training
    plt.figure(figsize=(12, 8))
    num_episodes = len(list(best_by_algo.values())[0])

    for key, best_scores in best_by_algo.items():
        print(key)
        algo, combination = key

        params = dict(combination)
        label = get_algo_str(algo) + ': '
        label += (
            f"γ={params['gamma']} "
            f"ε={params['epsilon']} "
        )
        if 'alpha' in params:
            label += f"α={params['alpha']} "

        if 'lambda' in params:
            label += f", λ={params['lambda']} "

        # Calcolo del passo per il sottocampionamento
        step = max(1, num_episodes // 10)  # Calcolo del passo per sottocampionamento
        x_values = range(1, num_episodes + 1, step)
        y_values = [best_scores[i - 1] for i in x_values]

        plt.plot(x_values, y_values, label=label)

    plt.xticks(range(1, num_episodes + 1, max(1, num_episodes // 10)))

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend(title="Algorithm with hyperparameters", loc="best")
    plt.savefig(filename)
    # plt.show()


def get_hyperparams(algo):
    if algo == 'mc':
        return [
            {"gamma": 0.2, "epsilon": 0.9},
            {"gamma": 0.5, "epsilon": 0.7},
            {"gamma": 0.8, "epsilon": 0.8},
        ]
    elif algo == 'sarsa' or algo =='ql':
        return [
            {"gamma": 0.2, "alpha": 0.2, "epsilon": 0.8},
            {"gamma": 0.8, "alpha": 0.3, "epsilon": 0.7},
            {"gamma": 0.5, "alpha": 0.4, "epsilon": 0.9},
        ]
    elif algo == 'sarsaL':
        return [
        {"gamma": 0.8, "alpha": 0.4, "epsilon": 0.7, "lambda": 0.3},
        {"gamma": 0.5, "alpha": 0.2, "epsilon": 0.8, "lambda": 0.5},
        {"gamma": 0.2, "alpha": 0.3, "epsilon": 0.9, "lambda": 0.9},
    ]
    else: return None

