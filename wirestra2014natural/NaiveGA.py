import random
import numpy as np

class Individual:
    def __init__(self):
        self.fitness = 0
        self.chromosome = None


class Population:
    def __init__(self):
        self.population = []

class GA:
    def __init__(self, population_size=100, mutation_rate=0.05, generations=20000, sigma=100, tournament_size=3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.sigma = sigma
        self.tournament_size = tournament_size
        self.up_bound = 50000

        self.population = []
        self.initialize_population()

    def evaluate_individual_fitness(self, an_individual):
        an_individual.fitness = self.evaluate_fitness(an_individual.chromosome)
        return an_individual.fitness

    def evaluate_fitness(self, chromosome):
        return -(chromosome - 10) ** 2 + 100

    def evaluate_population(self):
        for an_individual in self.population:
            self.evaluate_individual_fitness(an_individual)
        self.sort_population()

    def initialize_population(self):
        for i in range(self.population_size):
            an_individual = Individual()
            an_individual.chromosome = random.randint(-self.up_bound, self.up_bound)
            self.population.append(an_individual)
            self.evaluate_individual_fitness(an_individual)
        self.sort_population()

    def sort_population(self):
        self.population = sorted(self.population, key=lambda i: i.fitness, reverse=True)

    def mutate(self):
        fitness_sum = sum(x.fitness for x in self.population)
        avg_fitness = fitness_sum / float(self.population_size)

        # chromosome_sum = sum(x.chromosome for x in self.population)
        # avg_chromosome = chromosome_sum / float(self.population_size)
        for an_individual in self.population:
            if random.uniform(0, 1) < self.mutation_rate:
                # tmp = np.random.normal(avg_fitness, self.sigma)
                tmp = random.randint(-self.up_bound, self.up_bound)
                if self.evaluate_fitness(tmp) > an_individual.fitness:
                    an_individual.chromosome = tmp
        self.sort_population()

    def tournament(self):
        a_tournament = []
        for i in range(self.tournament_size):
            an_idx = random.randint(0, self.population_size-1)
            a_tournament.append(an_idx)
        return self.population[min(a_tournament)]

    def reproduce(self):
        new_generation = []
        for i in range(self.population_size):
            new_generation.append(self.tournament())
        self.population = new_generation
        self.sort_population()

    def evolve(self):
        for i in range(self.generations):
            self.evaluate_population()
            self.reproduce()
            self.mutate()
            print "generation: ", i, ", best individual fitness: ", self.population[0].fitness, ", its chromosome: ", \
                self.population[0].chromosome

