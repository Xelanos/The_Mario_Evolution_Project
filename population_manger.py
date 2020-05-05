import numpy as np

class Member():

    def __init__(self, genes, fitness_score=0):
        self.genes = genes
        self.fitness_core = 0
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

    def __iter__(self):
        return self.population.__iter__()


    def add_member(self, member):
        self.population.append(member)


    def remove_member(self, member):
        self.population.remove(member)

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


    def update_breeding_probability(self):
        fitness_sum = 0
        for member in self.population:
            fitness_sum += member.fitness
        for member in self.population:
            member.breeding_probability = member.fitness / fitness_sum

    def pick_from_population(self):
        i = 0
        r = np.random.uniform()
        while r > 0:
            r = r - self.population[i].breeding_probability
            i += 1

        return self.population[i - 1]

    def make_next_generation(self):
        # new_gen = MarioBasicPopulationManger(self.size)
        # for _ in range(self.size):
        #     new_gen.add_member(self.breed(self.pick_from_population(), self.pick_from_population()))
        return self


    def breed(self, first_member, second_member):
        pass

    def mutate(self, member):
        pass


