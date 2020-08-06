import copy
from player import MarioPlayer
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from itertools import combinations

DEFAULT_POPULATION_SIZE = 150
ELITE_DEFAULT_SIZE = 2


class Member():
    """
    Represent a member in PopulationManger
    """
    def __init__(self, genes, fitness_score=0, name=""):
        self.genes = genes
        self.fitness_score = fitness_score
        self.mating_probability = 0
        self.name = name

    def set_mating_probabilty(self, prob):
        if prob > 1 or prob < 0:
            raise Exception(f"Invalid probabily given: {prob}")

        self.mating_probability = prob

    def set_fitness_score(self, score):
        self.fitness_score = score

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name


class PopulationManger():

    def __init__(self, population_size):
        self.population = []
        self.size = population_size
        self.gen_number = 0

    def __iter__(self):
        return self.population.__iter__()

    def add_member(self, member):
        self.population.append(member)

    def remove_member(self, member):
        self.population.remove(member)
        self.size -= 1

    def pick_from_population(self):
        pass

    def mutate(self, member):
        pass

    def update_breeding_probability(self):
        pass

    def make_next_generation(self):
        pass

    def breed(self, first_member, second_member):
        pass


class MarioBasicPopulationManger(PopulationManger):

    def __init__(self, population_size=DEFAULT_POPULATION_SIZE, num_of_actions=len(SIMPLE_MOVEMENT),
                 elite_size=ELITE_DEFAULT_SIZE):
        super().__init__(population_size)
        self.num_of_actions = num_of_actions
        self.elite_size = min(max(2, elite_size), self.size)
        self.cross_prob = 0.5
        self.mutation_rate = 0.80 - (0.0006 * self.gen_number)
        self.mutation_power = 0.1
        self.tournament_size = 10
        # init all members:
        for index in range(self.size):
            player = MarioPlayer(self.num_of_actions)
            member_name = "member_{index}_gen_{gen_index}".format(index=index, gen_index=self.gen_number)
            self.add_member(Member(player.get_weights(), 0, member_name))

    def pick_from_population(self):
        tournament = np.random.choice(self.population, size=self.tournament_size)
        tournament = sorted(tournament, reverse=True, key=lambda member: member.fitness_score)
        i = np.random.geometric(0.8, size=1)[0]
        i -= 1  # because geometric is from 1 and we want from 0
        return tournament[i if i < self.tournament_size else self.tournament_size - 1]

    def make_next_generation(self):
        average_fitness = sum(member.fitness_score for member in self.population)/ len(self.population)
        print(f'Average fitness this gen : {average_fitness}')
        self.gen_number += 1
        elite = self.get_elite()
        print(f'Best fitness : {elite[0].fitness_score}')
        self.population = elite
        self.size = len(self.population)
        self.mutation_rate -= (0.0006 * self.gen_number)
        index = 0
        for parent1, parent2 in combinations(elite, 2):
            new_member1, new_member2 = self.breed(parent1, parent2)
            new_member1.set_name("member_{index}_gen_{gen_index}".format(index=index, gen_index=self.gen_number))
            self.add_member(new_member1)
            new_member2.set_name("member_{index}_gen_{gen_index}".format(index=index+1, gen_index=self.gen_number))
            self.add_member(new_member2)
            index += 2
            self.size += 2

    def get_elite(self):
        return sorted(self.population, key=lambda member: member.fitness_score, reverse=True)[:self.elite_size]

    def breed(self, first_member, second_member):
        new_weights1 = []
        new_weights2 = []
        cross_prob = 0.5
        mutation = np.random.uniform() < self.mutation_rate
        for weights1, weights2 in zip(first_member.genes, second_member.genes):
            new1 = copy.deepcopy(weights1)
            new2 = copy.deepcopy(weights2)
            i = np.random.rand(*weights1.shape) > cross_prob
            new1[i] = weights2[i]
            new2[i] = weights1[i]
            new_weights1.append(new1)
            new_weights2.append(new2)

        child1 = Member(new_weights1)
        child2 = Member(new_weights2)
        if mutation:
            self.mutate(child1)
            self.mutate(child2)
        return child1, child2

    def mutate(self, member):
        new_weights = []
        for weights in member.genes:
            i = np.random.rand(*weights.shape) < self.mutation_power
            noise = 2 * np.random.rand(*weights.shape) - 1
            weights[i] = noise[i]
            new_weights.append(weights)
        member.genes = new_weights



