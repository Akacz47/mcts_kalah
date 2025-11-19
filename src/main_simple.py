import numpy as np
from mcts import MCTS
from c4 import C4
from game_runner_cpu import GameRunner
from kalah import Kalah
 
if __name__=="__main__":
    AI = MCTS(search_time_limit=0.1, search_steps_limit=np.inf, vanilla=False)
    #game_runner = GameRunner(C4,None,AI,0,1,None)
    game_runner = GameRunner(Kalah,AI,AI,0,1,None) # None, AI
    outcome,game_info = game_runner.run()