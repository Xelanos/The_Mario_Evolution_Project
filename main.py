from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from mario_evolution import GeneticMario

INITIAL_POP = 150
GENERATIONS = 201
TIME_SCALE = 2000
DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'



actions = SIMPLE_MOVEMENT



if __name__ == '__main__':
    g = GeneticMario(DEFAULT_ENVIRONMENT, actions, GENERATIONS, INITIAL_POP)
    g.run(render_every=20)
