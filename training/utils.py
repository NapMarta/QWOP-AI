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


def plot_score(game_scores, title, filename):
    plt.figure(figsize=(12, 8))
    num_episodes = len(game_scores)
    plt.xticks(range(1, num_episodes + 1, max(1, num_episodes // 10)))

    for params_tuple, scores in game_scores.items():
        params = dict(params_tuple)
        label = (
            f"γ={params['gamma']}, "
            f"α={params['alpha']}, "
            f"ε={params['epsilon']}"
        )
    if 'lambda' in params:
        label += f", λ={params['lambda']}"

    plt.plot(scores, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend(title="Hyperparameters", loc="best")
    plt.savefig(filename)
    plt.show()


def plot_score_test(game_scores, title, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(game_scores, label="Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def get_hyperparams(algo):
    if algo == 'mc':
        return [
            {"gamma": 0.1, "epsilon": 0.2},
            {"gamma": 0.5, "epsilon": 0.3},
            {"gamma": 0.9, "epsilon": 0.1},
        ]
    elif algo == 'sarsa' or algo =='ql':
        return [
            {"gamma": 0.1, "alpha": 0.1, "epsilon": 0.7},
            {"gamma": 0.5, "alpha": 0.5, "epsilon": 0.8},
            {"gamma": 0.9, "alpha": 0.9, "epsilon": 0.9},
        ]
    elif algo == 'sarsaL':
        return [
        {"gamma": 0.1, "alpha": 0.1, "epsilon": 0.7, "lambda": 0.2},
        {"gamma": 0.5, "alpha": 0.5, "epsilon": 0.8, "lambda": 0.5},
        {"gamma": 0.9, "alpha": 0.9, "epsilon": 0.9, "lambda": 0.9},
    ]
    else: return None

