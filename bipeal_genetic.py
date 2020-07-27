import gc
import pickle
import traceback
from multiprocessing import Pool

import gym
gym.logger.set_level(40)
from gym.wrappers import monitor

from player import BipedalPlayer
from population_manger import *


class GeneticBipedal:

    def __init__(self, generations, initial_pop, time_scale):
        self.generations = generations
        self.inital_pop = initial_pop
        self.population = MarioBasicPopulationManger(self.inital_pop)
        self.elite = None
        self.generation = 0
        self.time_scale = time_scale
        self._init_pop()

    def _init_pop(self):
        weights = Pool(processes=1).map(self._init_population_player, range(self.inital_pop))
        for w in weights:
            self.population.add_member(Member(w, 0))

    def _init_population_player(self, i):
        p = BipedalPlayer()
        return p.get_weights()

    def run(self, render_every=100):
        try:
            for gen in range(self.generations):
                self.generation = gen
                print(f'Staring generation {gen + 1}')
                self.render = (gen % render_every == 0)
                if gen == 0: self.render = False
                pool = Pool()
                members = pool.map_async(self.run_player, self.population).get()
                pool.close()
                pool.join()
                for member in members:
                    self.population.add_member(member)
                self.population = self.population.make_next_generation()
                print('Generation Done\n')
                gc.collect()
            self._save()
        except Exception as e:
            self._save()
            traceback.print_exc(e.__str__())
            return

    def run_player(self, member, record=False, render=True):
        env = gym.make('BipedalWalkerHardcore-v3')
        player = BipedalPlayer(member.genes)

        if record:
            rec = monitor.video_recorder.VideoRecorder(env, path=f"vid/gen.mp4")
        env.reset()
        done = False
        action = [0, 0, 0, 0]

        for step in range(self.time_scale):
            if done:
                break
            state, reward, done, info = env.step(action)
            if record:
                rec.capture_frame()
            action = player.act(state)
            player.reward += reward
            if self.render:
                env.render()
        if record:
            rec.close()
        env.close()
        return Member(player.get_weights(), player.calculate_fittness())

    def _save(self):

        with open("saved/saved_gen.pic", 'wb') as f:
            pickle.dump(self.generation, f)

        with open("saved/saved_pop.pic", 'wb') as f:
            pickle.dump(self.population, f)
