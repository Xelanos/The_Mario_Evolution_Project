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

TIME_SCALE = 1500

class GeneticMario:

    def __init__(self, actions, generations, initial_pop):
        self.actions = actions
        self.num_of_actions = len(actions)
        self.generations = generations
        self.inital_pop = initial_pop
        self.population = MarioBasicPopulationManger(self.inital_pop)
        self.elite = None
        self.generation = 0
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
            rec = monitor.video_recorder.VideoRecorder(env, path=f"vid/gen.mp4")
        env.reset()
        done = False
        action = 0

        for step in range(TIME_SCALE):
            if done:
                break
            state, reward, done, info = env.step(action)
            if record:
                rec.capture_frame()
            action = player.act(state)
            player.update_info(info)
            player.reward += reward
            if info['life'] < 2: # will repeat death, so why try more
                done = True
            if self.render:
                env.render()
        if record:
            rec.close()
        env.close()
        return Member(player.get_weights(), player.calculate_fittness())



    def update_fitness_and_find_elite(self):
        best_fit = 0
        for player in self.poplation:
            fit = player.calculate_fittness()
            if fit > best_fit:
                self.elite = player


    def _init_pop(self):
        weights = Pool(processes=1).map(self._init_population_player, range(self.inital_pop))
        for w in weights:
            self.population.add_member(Member(w, 0))


    def _init_population_player(self, i):
        p = MarioPlayer(self.num_of_actions)
        return p.get_weights()


    def crossover(self, first_parent, second_parent):
        first_parent_kernel = first_parent.model.get_layer('hidden').get_weights()[0]
        first_parent_bias = first_parent.model.get_layer('hidden').get_weights()[1]

        second_parent_kernel = second_parent.model.get_layer('hidden').get_weights()[0]
        second_parent_bias = second_parent.model.get_layer('hidden').get_weights()[1]

        cross_prob = first_parent.breeding_probability / (first_parent.breeding_probability + second_parent.breeding_probability)
        wieghts_len = len(first_parent_kernel[0])
        idx_to_take_from_second = np.random.uniform(size=wieghts_len) > cross_prob

        child_kernel = np.copy(first_parent_kernel)
        i = np.random.rand(*first_parent_kernel.shape) > cross_prob
        child_kernel[i] = second_parent_kernel[i]

        child_bias = np.copy(first_parent_bias)
        child_bias[idx_to_take_from_second] = second_parent_bias[idx_to_take_from_second]

        return [child_kernel, child_bias]

    def _save(self):
        if not os.path.isdir("saved"):
            os.mkdir("saved")

        with open("saved/saved_gen.pic", 'wb') as f:
            pickle.dump(self.generation, f)


        with open("saved/saved_pop.pic", 'wb') as f:
            pickle.dump(self.population, f)

