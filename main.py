from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from mario_evolution import GeneticMario
from player import MarioPlayer
from population_manger import Member

import numpy as np


INITIAL_POP = 101
ELITE_DEFAULT_SIZE = 10
GENERATIONS = 201
TIME_SCALE = 2000



actions = COMPLEX_MOVEMENT



if __name__ == '__main__':


    loded = np.load("best_model.npz")
    best_wieghts = []
    for i in range(10):
        best_wieghts.append(loded[f'arr_{i}'])


    genetic_1_1 = GeneticMario('SuperMarioBros-1-1-v0', actions, GENERATIONS, INITIAL_POP, ELITE_DEFAULT_SIZE)
    best_player = Member(best_wieghts, name="Best1-1")
    genetic_1_1.render = True
    genetic_1_1.run_player(best_player)
    
    genetic_3_2 = GeneticMario('SuperMarioBros-3-2-v0', actions, GENERATIONS, INITIAL_POP, ELITE_DEFAULT_SIZE)
    best_player = Member(best_wieghts, name="Best3-2")
    genetic_3_2.render = True
    genetic_3_2.run_player(best_player)


