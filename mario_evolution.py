from player import MarioPlayer
from population_manger import *

import gym.wrappers.monitor as monitor

import traceback
import pickle
from pandas import DataFrame

import os
import gc

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from multiprocessing import Pool, cpu_count

TIME_SCALE = 200
INITIAL_LIFE = 2
NO_ADVANCE_STEP_LIMIT = 100


class GeneticMario:

    def __init__(self, mario_environment, actions, generations, initial_pop, steps_scale=TIME_SCALE, allow_death=False,
                 standing_steps_limit=NO_ADVANCE_STEP_LIMIT, output_dir=""):
        self.actions = actions
        self.num_of_actions = len(actions)
        self.generations = generations
        self.initial_pop = initial_pop
        self.population = MarioBasicPopulationManger(self.initial_pop, self.num_of_actions)
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
        try:
            for gen in range(self.generations):
                self.generation = gen
                print(f'Staring generation {gen + 1}')
                self.current_gen_output_dir = os.path.join(self.output_dir, "gen_{}".format(gen+1))
                os.mkdir(self.current_gen_output_dir)

                self.render = (gen % render_every == 0) if render_every else False
                self.record = (gen % record_every == 0) if record_every else False
                if self.record:
                    os.mkdir(os.path.join(self.current_gen_output_dir, "vid"))

                pool = Pool()
                gen_outcomes_and_updated_members = pool.map_async(self.run_player, self.population).get()
                pool.close()
                pool.join()
                gen_outcomes = []
                updated_members = []
                for outcome, member in gen_outcomes_and_updated_members:
                    gen_outcomes.append(outcome)
                    outcomes.append(outcome)
                    updated_members.append(member)
                self.population.population = updated_members
                self._save_generation_outcome(gen_outcomes)
                self.population = self.population.make_next_generation()
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
        player = MarioPlayer(self.num_of_actions, member.genes)

        if self.record:
            rec_output_path = os.path.join(self.current_gen_output_dir, "vid", "{name}.mp4".
                                           format(name=member.get_name()))
            rec = monitor.video_recorder.VideoRecorder(env, path=rec_output_path)

        env.reset()
        done = False
        state = None

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
        return outcome, member

    def _save_generation_outcome(self, outcomes):
        df = DataFrame(outcomes)
        df.to_csv(os.path.join(self.current_gen_output_dir, "gen_{}_output.csv".format(self.generation)))
        with open(os.path.join(self.current_gen_output_dir, "PopulationManger.pic"), 'wb') as f:
            pickle.dump(self.population, f)

    def _save(self, outcomes):
        df = DataFrame(outcomes)
        df.to_csv(os.path.join(self.output_dir, "genetic_output.csv"))
        with open(os.path.join(self.output_dir, "PopulationManger.pic"), 'wb') as f:
            pickle.dump(self.population, f)



