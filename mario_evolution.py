from player import MarioPlayer
from population_manger import *

import gym.wrappers.monitor as monitor

import traceback
import pickle

import os
import gc

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from multiprocessing import Pool, cpu_count

TIME_SCALE = 200
INITIAL_LIFE = 2
NO_ADVANCE_STEP_LIMIT = 100


class GeneticMario:

    def __init__(self, actions, generations, initial_pop, steps_scale=TIME_SCALE, allow_death=False,
                 standing_steps_limit = NO_ADVANCE_STEP_LIMIT):
        self.actions = actions
        self.num_of_actions = len(actions)
        self.generations = generations
        self.inital_pop = initial_pop
        self.population = MarioBasicPopulationManger(self.inital_pop)
        self.elite = None
        self.generation = 0
        self.steps_scale = steps_scale
        self.allow_death = allow_death
        self.standing_steps_limit = standing_steps_limit
        self._init_pop()

    def run(self, render_every=100):
        try:
            for gen in range(self.generations):
                self.generation = gen
                print(f'Staring generation {gen + 1}')
                if render_every:
                    self.render = (gen % render_every == 0)
                else:
                    self.render = False
                pool = Pool()
                members = pool.map_async(self.run_player, self.population).get()
                pool.close()
                pool.join()
                for member in members:
                    self.population.add_member(member)
                self.population = self.population.make_next_generation()
                gc.collect()
            self._save()
        except Exception as e:
            self._save()
            traceback.print_exc(e)
            return

    def run_player(self, member, record=False, render=True):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, self.actions)
        player = MarioPlayer(self.num_of_actions, member.genes)

        if record:
            if not os.path.isdir("vid"):
                os.mkdir("vid")
            rec = monitor.video_recorder.VideoRecorder(env, path=f"vid/gen.mp4")
        env.reset()
        done = False
        action = 0

        last_x_pos = 0
        same_x_pos_cunt = 0

        for step in range(self.steps_scale):
            if done:
                break
            state, reward, done, info = env.step(action)
            if record:
                rec.capture_frame()
            action = player.act(state)
            player.update_info(info)
            player.reward += reward
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
            if self.render:
                env.render()
        if record:
            rec.close()
        env.close()
        return Member(player.get_weights(), player.calculate_fittness())


    def _init_pop(self):
        weights = Pool(processes=1).map(self._init_population_player, range(self.inital_pop))
        for w in weights:
            self.population.add_member(Member(w, 0))


    def _init_population_player(self, i):
        p = MarioPlayer(self.num_of_actions)
        return p.get_weights()

    def _save(self):
        if not os.path.isdir("saved"):
            os.mkdir("saved")

        with open("saved/saved_gen.pic", 'wb') as f:
            pickle.dump(self.generation, f)

        with open("saved/saved_pop.pic", 'wb') as f:
            pickle.dump(self.population, f)

