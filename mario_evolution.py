from population_manger import *

import gym.wrappers.monitor as monitor
from wrappers import WarpFrame

import traceback
import time
from pandas import DataFrame

import os
import gc

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

ELITE_DEFAULT_SIZE = 10
TIME_SCALE = 200
INITIAL_LIFE = 2
NO_ADVANCE_STEP_LIMIT = 100
SAVE_POPULATION_MANAGER_EVERY_GENERATION = False


class GeneticMario:

    def __init__(self, mario_environment, actions, generations, initial_pop, elite_size=ELITE_DEFAULT_SIZE,
                 steps_scale=TIME_SCALE, allow_death=False, standing_steps_limit=NO_ADVANCE_STEP_LIMIT, output_dir=""):
        self.actions = actions
        self.num_of_actions = len(actions)
        self.generations = generations
        self.initial_pop = initial_pop
        self.elite_size = elite_size
        self.population = MarioBasicPopulationManger(self.initial_pop, self.num_of_actions, self.elite_size)
        self.elite = None
        self.generation = 0
        self.steps_scale = steps_scale
        self.allow_death = allow_death
        self.standing_steps_limit = standing_steps_limit
        self.record = False
        self.render = False
        self.output_dir = output_dir
        self.current_gen_output_dir = output_dir
        self.env = mario_environment

    def run(self, render_every=100, record_every=0):
        outcomes = []
        print("Initializing the first generation with random weights.")
        self.population.init_pop()
        try:
            for gen in range(self.generations):
                self.generation = gen
                print(f'Staring generation {gen + 1}')
                t = time.time()
                self.current_gen_output_dir = os.path.join(self.output_dir, "gen_{}".format(gen+1))
                if not os.path.isdir(self.current_gen_output_dir):
                    os.mkdir(self.current_gen_output_dir)

                self.render = (gen % render_every == 0) if render_every else False
                self.record = (gen % record_every == 0) if record_every else False
                if self.record:
                    if not os.path.isdir(os.path.join(self.current_gen_output_dir, "vid")):
                        os.mkdir(os.path.join(self.current_gen_output_dir, "vid"))
                gen_outcomes = []
                for member in self.population:
                    outcome = self.run_player(member)
                    gen_outcomes.append(outcome)
                    outcomes.append(outcome)
                self._save_generation_outcome(gen_outcomes)
                print(f"finish {gen + 1} in {time.time() - t}")
                if gen != self.generations - 1:
                    self.population.make_next_generation()
                gc.collect()

            self._save(outcomes)
            return outcomes
        except Exception as e:
            self._save(outcomes)
            traceback.print_exc(e)
            return outcomes

    def run_player(self, member):
        env = gym_super_mario_bros.make(self.env)
        env = JoypadSpace(env, self.actions)
        env = WarpFrame(env)
        player = MarioPlayer(self.num_of_actions, member.genes)

        if self.record:
            rec_output_path = os.path.join(self.current_gen_output_dir, "vid", "{name}.mp4".
                                           format(name=member.get_name()))
            rec = monitor.video_recorder.VideoRecorder(env, path=rec_output_path)

        state = env.reset()
        done = False

        last_x_pos = 0
        same_x_pos_cunt = 0

        for step in range(self.steps_scale):
            if done:
                break
            action = player.act(state)
            state, reward, done, info = env.step(action)

            if self.record:
                rec.capture_frame()
            if self.render:
                env.render()

            player.update_info(info)
            player.update_reward(reward)
            if last_x_pos == info['x_pos']:
                same_x_pos_cunt += 1
            else:
                same_x_pos_cunt = 0
                last_x_pos = info['x_pos']
            if same_x_pos_cunt > self.standing_steps_limit:  # end the run if player don't advance:
                done = True
            if not self.allow_death and info['life'] < INITIAL_LIFE:  # will repeat death, so why try more
                done = True
            if info['flag_get']:  # if got to the flag - run is ended.
                done = True

        if self.record:
            rec.close()
        env.close()
        member.set_fitness_score(player.calculate_fitness())
        outcome = player.get_run_info()
        outcome['generation'] = self.generation
        outcome['index'] = member.get_name()
        return outcome

    def _save_generation_outcome(self, outcomes):
        df = DataFrame(outcomes)
        df.to_csv(os.path.join(self.current_gen_output_dir, "gen_{}_output.csv".format(self.generation)))
        if SAVE_POPULATION_MANAGER_EVERY_GENERATION:
            self.population.save_population(self.current_gen_output_dir)

    def _save(self, outcomes):
        df = DataFrame(outcomes)
        df.to_csv(os.path.join(self.output_dir, "genetic_output.csv"))
        self.population.save_population(self.output_dir)



