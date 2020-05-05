from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


from mario_evolution import GeneticMario

INITIAL_POP = 21
GENERATIONS = 201

actions = SIMPLE_MOVEMENT




if __name__ == '__main__':


    g = GeneticMario(actions, GENERATIONS, INITIAL_POP)
    g.run(render=False)
