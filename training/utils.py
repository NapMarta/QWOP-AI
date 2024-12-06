
import qwop_gym
import gymnasium as gym

def get_init_env():

    browser_path, driver_path = _get_path()

    env = gym.make(
        "QWOP-v1",
        browser = browser_path,
        driver = driver_path,
        auto_draw = True,
        stat_in_browser = True,
        reduced_action_set = True
    )

    return env

def _get_path():
    file_path = "config_env.txt"
    browser_path = ""
    driver_path = ""

    try:
        with open(file_path, 'r') as file:
            righe = file.readlines()
            browser_path = righe[1].strip()  
            driver_path = righe[3].strip()  

    except FileNotFoundError:
        print(f"Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    print("Browser Path:", browser_path)
    print("Driver Path:", driver_path)

    return browser_path, driver_path
