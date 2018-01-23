import random
import numpy as np
import tensorflow as tf

class Individual:
    def __init__(self):
        self.fitness = 0
        self.chromosome = None

class NESGA:
    def __init__(self, population_size=5, mutation_rate=0.05, generations=10, sigma=1000, tournament_size=3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.sigma = sigma
        self.tournament_size = tournament_size
        self.up_bound = 20000

        self.population = np.zeros(shape=(self.population_size, 1))
        self.initialize_population()

    def evaluate_fitness(self, chromosome):
        return -(chromosome - 10) ** 2 + 100

    def evaluate_population(self):
        rtn = np.zeros(shape=(self.population_size, 1))
        for i in range(self.population_size):
            rtn[i] = self.evaluate_fitness(self.population[i])
        self.sort_population()
        return rtn

    def initialize_population(self):
        for i in range(self.population_size):
            an_individual_chromosome = random.randint(-self.up_bound, self.up_bound)
            self.population[i] = an_individual_chromosome
        self.sort_population()

    def sort_population(self):
        self.population = sorted(self.population, key=lambda x: self.evaluate_fitness(x), reverse=True)

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

    def get_parameters_of_density(self, chromosomes):
        tmp_dict = {}
        rtn = np.zeros(shape=(self.population_size, 1))

        for a_chromosome in chromosomes:
            if tmp_dict.has_key(a_chromosome):
                pass
            else:
                tmp_dict[a_chromosome] = len(x==a_chromosome for x in chromosomes) / self.population_size

        for i in range(self.population_size):
            rtn[i] = tmp_dict[chromosomes[i]]

    def evolve(self):
        alpha = 0.05
        tf_population = tf.placeholder(tf.float32)
        loss = tf.reduce_sum((-tf.pow((tf_population - 5), 2) + 100) * self.get_parameters_of_density(tf_population))
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # for i in range(self.generations):
        #     sess.run(feed_dict={})

