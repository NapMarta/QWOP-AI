from threading import Thread
import main_training
from threading import Lock


class ThreadAlgo(Thread):
    def __init__(self, algo, gamma, alpha, eps, lam, end_step, results):
        super().__init__()
        self.algo = algo
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.end_step = end_step
        self.results = results

    def run(self):
        
        lock = Lock()

        print(f"\n\n#### Execute {self.algo} ####")
        tmp_res = main_training.worker(self.algo, self.gamma, self.alpha, self.eps, self.lam, self.end_step)
        
        with lock:
            self.results[self.algo] = tmp_res