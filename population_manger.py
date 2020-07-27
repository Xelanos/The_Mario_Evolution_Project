import copy

import numpy as np

class Member():

    def __init__(self, genes, fitness_score=0):
        self.genes = genes
        self.fitness_score = fitness_score
        self.mating_probabilty = 0

    def set_mating_probabilty(self, prob):
        if prob > 1 or prob < 0:
            raise Exception(f"Invalid probabily given: {prob}")

        self.mating_probabilty = prob

    def set_fitness_score(self, score):
        if score < 0:
            raise Exception(f"Fitness score cannot be smaller then 0")
        self.fitness_score = score



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

    def __init__(self, population_size):
        super().__init__(population_size)
        self.cross_prob = 0.5
        self.mutation_rate = 0.80 - (0.0006 * self.gen_number)
        self.mutation_power = 0.1
        self.tournament_size = 10

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
        new_gen = MarioBasicPopulationManger(self.size)
        elite = self.find_elite()
        print(f'Best fitness : {elite.fitness_score}')
        new_gen.add_member(elite)
        for _ in range(self.size - 1):
            parent1 = self.pick_from_population()
            parent2 = self.pick_from_population()
            while parent1 == parent2:
                parent2 = self.pick_from_population()
            new_gen.add_member(self.breed(parent1, parent2))
        return new_gen

    def find_elite(self):
        return sorted(self.population, key=lambda member: member.fitness_score, reverse=True)[0]

    def breed(self, first_member, second_member):
        new_weights = []
        cross_prob = 0.5
        mutation = np.random.uniform() < self.mutation_rate
        for weights1, weights2 in zip(first_member.genes, second_member.genes):
            new = copy.deepcopy(weights1)
            i = np.random.rand(*weights1.shape) > cross_prob
            new[i] = weights2[i]
            if mutation:
                i = np.random.rand(*weights1.shape) < self.mutation_power
                noise = 2 * np.random.rand(*weights1.shape) - 1
                new[i] = noise[i]

            new_weights.append(new)
        return Member(new_weights)

    def mutate(self, member):
        pass


