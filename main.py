from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import gym.wrappers.monitor as monitor
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)

from mario_evolution import GeneticMario
import tensorflow as tf
import numpy as np

INITIAL_POP = 10
GENERATIONS = 10


g = GeneticMario(env, GENERATIONS, INITIAL_POP)
g.run()





if __name__ == '__main__':
    pass