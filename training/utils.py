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


def plot_value_function(agents, filename, title="Value Function"):
    V = get_actionValue(agents.q_values)

    # Min e Max per la defiizione degli assi
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z = np.apply_along_axis(lambda _: V.get((_[0], _[1]), 0.0), 2, np.dstack([X, Y]))

    # Plot della value function
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)

    plt.savefig(filename)
    plt.show()



def plot_score(game_scores, title, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(game_scores, label="Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def save_model(Q, filename):
    q_table_str_keys = {str(key): value for key, value in Q.items()}

    with open(filename, "w") as file:
        json.dump(q_table_str_keys, file)


def load_model(filename):
    with open(filename, "r") as file:
        q_table_str_keys = json.load(file)

    Q = {eval(key): value for key, value in q_table_str_keys.items()}
    
    return Q


