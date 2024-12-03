
import qwop_gym
import gymnasium as gym

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