
import numpy as np
from agent import Agent
import qwop_gym
import gymnasium as gym
from collections import defaultdict

class MCAgent(Agent):
    def __init__(self, env, gamma=1.0, alpha=0, eps=0.1):
        # il fattore alpha non serve?
        super().__init__(env, gamma, alpha, eps)


    
    def _epsilon_soft_policy(self, state):
        pass
    
    

    


def get_init_env():
    env = gym.make(
        "QWOP-v1",
        browser="C:/Program Files/Google/Chrome/Application/chrome.exe",
        driver="C:/Program Files (x86)/chromedriver-win64/chromedriver.exe",
        auto_draw=True,
        stat_in_browser=True,
        reduced_action_set=True
    )

    state = env.reset()
    return env


if __name__ == "__main__":
    env = get_init_env()
    
    # Create a new agent
    agentMC = MCAgent(env= env)



