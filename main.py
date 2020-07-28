from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from bipeal_genetic import GeneticBipedal
from mario_evolution import GeneticMario

INITIAL_POP = 150
GENERATIONS = 2001
TIME_SCALE = 1500



actions = SIMPLE_MOVEMENT



if __name__ == '__main__':

    # g = GeneticMario(actions, GENERATIONS, INITIAL_POP)
    g = GeneticBipedal(GENERATIONS, INITIAL_POP, TIME_SCALE)
    g.run(render_every=20)
