import numpy as np

class Member():

    def __init__(self, genes, fitness_score=1):
        self.genes = genes
        self.fitness_core = fitness_score
        self.mating_probabilty = 0

    def set_mating_probabilty(self, prob):
        if prob > 1 or prob < 0:
            raise Exception(f"Invalid probabily given: {prob}")

        self.mating_probabilty = prob

    def set_fitness_score(self, score):
        if score < 0:
            raise Exception(f"Fitness score cannot be smaller then 0")
        self.fitness_core = score



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
        self.mutation_rate = 0.40
        self.mutation_power = 0.70


    def update_breeding_probability(self):
        fitness_sum = 0
        for member in self.population:
            fitness_sum += member.fitness_core
        for member in self.population:
            member.mating_probabilty = member.fitness_core / fitness_sum

        print(f'average fitness this generation : {fitness_sum/ self.size}')

    def pick_from_population(self):
        i = 0
        r = np.random.uniform()
        while r > 0:
            r = r - self.population[i].mating_probabilty
            i += 1

        return self.population[i - 1]

    def make_next_generation(self):
        self.gen_number += 1
        new_gen = MarioBasicPopulationManger(self.size)
        new_gen.add_member(self.find_elite())
        self.update_breeding_probability()
        for _ in range(self.size - 1):
            parent1 = self.pick_from_population()
            parent2 = self.pick_from_population()
            while parent1 == parent2:
                parent2 = self.pick_from_population()
            new_gen.add_member(self.breed(parent1, parent2))
        return new_gen

    def find_elite(self):
        best_fit = 0
        elite = self.population[0]
        for memeber in self.population:
            fit = memeber.fitness_core
            if fit > best_fit:
                best_fit = fit
                elite = memeber
        return elite

    def breed(self, first_member, second_member):
        new_weights = []
        cross_prob = first_member.mating_probabilty / (
                first_member.mating_probabilty + second_member.mating_probabilty)
        mutation = np.random.uniform() < self.mutation_rate
        for weights1, weights2 in zip(first_member.genes, second_member.genes):
            new = np.copy(weights1)
            i = np.random.rand(*weights1.shape) > cross_prob
            new[i] = weights2[i]
            if mutation:
                i = np.random.rand(*weights1.shape) < self.mutation_power
                if len(weights1.shape) < 2:
                    noise = np.random.normal(0, 1, size=weights1.shape[0])
                else:
                    noise = np.random.multivariate_normal(np.ones(weights1.shape[1]), np.eye(weights1.shape[1]), size=weights1.shape[0])
                new[i] += noise[i]

            new_weights.append(new)
        return Member(new_weights)

    def mutate(self, member):
        pass


