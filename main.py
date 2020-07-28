from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from mario_evolution import GeneticMario

INITIAL_POP = 9
GENERATIONS = 2001

actions = SIMPLE_MOVEMENT

TIME_SCALE = 3000




if __name__ == '__main__':


    g = GeneticMario(actions, GENERATIONS, INITIAL_POP)
    g.run(render_every=0)
