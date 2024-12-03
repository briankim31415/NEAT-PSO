import random
import pickle
import copy
import itertools
import math
import multiprocessing as mp

from activations import *
from hyperparameters import Hyperparameters
from genome import *


class Specie(object):
    print('adding new code')
    """A specie represents a set of genomes whose genomic distances 
    between them fall under the Brain's delta threshold.
    """
    def __init__(self, max_fitness_history, *members):
        self._members = list(members)
        self._fitness_history = []
        self._fitness_sum = 0
        self._max_fitness_history = max_fitness_history

    def breed(self, mutation_probabilities, breed_probabilities):
        """Return a child as a result of either a mutated clone
        or crossover between two parent genomes.
        """
        # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities= [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child = random.choice(self._members).clone()
            child.mutate(mutation_probabilities)
        elif choice == "sexual":
            (mom, dad) = random.sample(self._members, 2)
            # shouldn't need to change this; handle convolutions + FC crossover in genomic_crossover
            child = genomic_crossover(mom, dad)

        return child

    def update_fitness(self):
        """Update the adjusted fitness values of each genome 
        and the historical fitness."""
        for g in self._members:
            g._adjusted_fitness = g._fitness / len(self._members)

        self._fitness_sum = sum([g._adjusted_fitness for g in self._members])
        self._fitness_history.append(self._fitness_sum)
        if len(self._fitness_history) > self._max_fitness_history:
            self._fitness_history.pop(0)

    def eliminate_weakest(self, fittest_only):
        """Exterminate the weakest genomes per specie."""
        self._members.sort(key=lambda g: g._fitness, reverse=True)
        if fittest_only:
            # Only keep the winning genome
            remaining = 1
        else:
            # Keep top 25%
            remaining = int(math.ceil(0.25*len(self._members)))

        self._members = self._members[:remaining]

    def get_best(self):
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)

    def can_progress(self):
        """Determine whether species should survive the elimination."""
        n = len(self._fitness_history)
        avg = sum(self._fitness_history) / n
        return avg > self._fitness_history[0] or n < self._max_fitness_history
    

class ConvSpecie(Specie):
    def __init__(self, max_fitness_history, *members):
        super().__init__(max_fitness_history, *members)

    def breed(self, mutation_probabilities, breed_probabilities):

         # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities= [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child = random.choice(self._members).clone()
            child.mutate(mutation_probabilities)
        elif choice == "sexual":
            (mom, dad) = random.sample(self._members, 2)
            # shouldn't need to change this; handle convolutions + FC crossover in genomic_crossover
            child = genomic_crossover(mom, dad)

        return child
    
        pass
        raise NotImplementedError("need to implement breed for convolutions")


class Brain(object):
    """Base class for a 'brain' that learns through the evolution
    of a population of genomes.
    """
    def __init__(self, inputs, outputs, population=100, hyperparams=Hyperparameters()):
        self._inputs = inputs
        self._outputs = outputs

        self._species = []
        self._population = population

        # Hyper-parameters
        self._hyperparams = hyperparams
        
        self._generation = 0
        self._current_species = 0
        self._current_genome = 0

        self._global_best = None

    def generate(self):
        """Generate the initial population of genomes."""
        for i in range(self._population):
            g = NeuralNetGenome(self._inputs, self._outputs, 
                       self._hyperparams.default_activation)
            g.generate()
            self.classify_genome(g)
        
        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def classify_genome(self, genome):
        """Classify genomes into species via the genomic
        distance algorithm.
        """
        if not self._species:
            # Empty population
            self._species.append(Specie(
                    self._hyperparams.max_fitness_history, genome
                )
            )
        else:
            # Compare genome against representative s[0] in each specie
            for s in self._species:
                rep =  s._members[0]
                dist = genomic_distance(
                    genome, rep, self._hyperparams.distance_weights
                )
                if dist <= self._hyperparams.delta_threshold:
                    s._members.append(genome)
                    return

            # Doesn't fit with any other specie, create a new one
            self._species.append(Specie(
                    self._hyperparams.max_fitness_history, genome
                )
            )

    def update_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)

        if current_top._fitness > self._global_best._fitness:
            self._global_best = current_top.clone()

    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        global_fitness_sum = 0
        for s in self._species:
            s.update_fitness()
            global_fitness_sum += s._fitness_sum

        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for s in self._species:
                for g in s._members:
                    g.mutate(self._hyperparams.mutation_probabilities)
        else:
            # Only keep the species with potential to improve
            surviving_species = []
            for s in self._species:
                if s.can_progress():
                    surviving_species.append(s)
            self._species = surviving_species

            # Eliminate lowest performing genomes per specie
            for s in self._species:
                s.eliminate_weakest(False)

            # Repopulate
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                diff = self._population - self.get_population()
                offspring = int(round(ratio * diff))
                for j in range(offspring):
                    self.classify_genome(
                        s.breed(
                            self._hyperparams.mutation_probabilities, 
                            self._hyperparams.breed_probabilities
                        )
                    )

            # No species survived
            # Repopulate using mutated minimal structures and global best
            if not self._species:
                for i in range(self._population):
                    if i%3 == 0:
                        g = self._global_best.clone()
                    else:
                        g = NeuralNetGenome(self._inputs, self._outputs, 
                                   self._hyperparams.default_activation)
                        g.generate()
                    g.mutate(self._hyperparams.mutation_probabilities)
                    self.classify_genome(g)

        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness <= self._hyperparams.max_fitness
        end = self._generation != self._hyperparams.max_generations

        return fit and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        s = self._species[self._current_species]
        if self._current_genome < len(s._members)-1:
            self._current_genome += 1
        else:
            if self._current_species < len(self._species)-1:
                self._current_species += 1
                self._current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self._current_species = 0
                self._current_genome = 0

    def evaluate_parallel(self, evaluator, *args, **kwargs):
        """Evaluate the entire population on separate processes
        to progress training. The evaluator function must take a Genome
        as its first parameter and return a numerical fitness score.

        Any global state passed to the evaluator is copied and will not
        be modified at the parent process.
        """
        max_proc = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_proc)
        
        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator, 
                    args=[self._species[i]._members[j]]+list(args), 
                    kwds=kwargs
                )

        for key in results:
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key].get())

        pool.close()
        pool.join()
        self.evolve()

    def get_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """Return the true (calculated) population size."""
        return sum([len(s._members) for s in self._species])

    def get_current(self):
        """Get the current genome for evaluation."""
        s = self._species[self._current_species]
        return s._members[self._current_genome]

    def get_current_species(self):
        """Get index of current species being evaluated."""
        return self._current_species

    def get_current_genome(self):
        """Get index of current genome being evaluated."""
        return self._current_genome

    def get_generation(self):
        """Get the current generation number of this population."""
        return self._generation

    def get_species(self):
        """Get the list of species and their respective member genomes."""
        return self._species

    def save(self, filename):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)
        

class ConvBrain(object):
    """Class for a convolutional'brain' that learns through the evolution
    of a population of genomes.
    """
    def __init__(self, conv_inputs, conv_outputs, dense_inputs, dense_outputs, population=100, hyperparams=Hyperparameters()):
        self._conv_inputs = conv_inputs
        self._conv_outputs = conv_outputs
        self._dense_inputs = dense_inputs
        self._dense_outputs = dense_outputs

        self._species = []
        self._population = population

        # Hyper-parameters
        self._hyperparams = hyperparams
        
        self._generation = 0
        self._current_species = 0
        self._current_genome = 0

        self._global_best = None

    def generate(self):
        """Generate the initial population of genomes."""
        for i in range(self._population):
            g = ConvolutionalGenome(self._conv_inputs, self._conv_outputs, self._dense_inputs, 
                                    self._dense_outputs, self._hyperparams.default_activation, 
                                    self._hyperparams.default_activation)
            g.generate()
            self.classify_genome(g)
        
        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def classify_genome(self, genome):
        """Classify genomes into species via the genomic
        distance algorithm.
        """
        if not self._species:
            # Empty population
            self._species.append(
                ConvSpecie(self._hyperparams.max_fitness_history, genome)
            )
        else:
            # Compare genome against representative s[0] in each specie
            for s in self._species:
                rep =  s._members[0]
                dist = conv_genomic_distance(
                    genome, rep, self._hyperparams.distance_weights
                )
                if dist <= self._hyperparams.delta_threshold:
                    s._members.append(genome)
                    return

            # Doesn't fit with any other specie, create a new one
            self._species.append(ConvSpecie(
                    self._hyperparams.max_fitness_history, genome
                )
            )

    def update_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)

        if current_top._fitness > self._global_best._fitness:
            self._global_best = current_top.clone()

    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        global_fitness_sum = 0
        for s in self._species:
            s.update_fitness()
            global_fitness_sum += s._fitness_sum

        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for s in self._species:
                for g in s._members:
                    g.mutate(self._hyperparams.mutation_probabilities)
        else:
            # Only keep the species with potential to improve
            surviving_species = []
            for s in self._species:
                if s.can_progress():
                    surviving_species.append(s)
            self._species = surviving_species

            # Eliminate lowest performing genomes per specie
            for s in self._species:
                s.eliminate_weakest(False)

            # Repopulate
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                diff = self._population - self.get_population()
                offspring = int(round(ratio * diff))
                for j in range(offspring):
                    self.classify_genome(
                        s.breed(
                            self._hyperparams.mutation_probabilities, 
                            self._hyperparams.breed_probabilities
                        )
                    )

            # No species survived
            # Repopulate using mutated minimal structures and global best
            if not self._species:
                for i in range(self._population):
                    if i%3 == 0:
                        g = self._global_best.clone()
                    else:
                        g = ConvolutionalGenome(self._conv_inputs, self._conv_outputs, self._dense_inputs, 
                                    self._dense_outputs, self._hyperparams.default_activation, 
                                    self._hyperparams.default_activation)
                        g.generate()
                    g.mutate(self._hyperparams.mutation_probabilities)
                    self.classify_genome(g)

        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness <= self._hyperparams.max_fitness
        end = self._generation != self._hyperparams.max_generations

        return fit and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        s = self._species[self._current_species]
        if self._current_genome < len(s._members)-1:
            self._current_genome += 1
        else:
            if self._current_species < len(self._species)-1:
                self._current_species += 1
                self._current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self._current_species = 0
                self._current_genome = 0

    def evaluate_parallel(self, evaluator, *args, **kwargs):
        """Evaluate the entire population on separate processes
        to progress training. The evaluator function must take a Genome
        as its first parameter and return a numerical fitness score.

        Any global state passed to the evaluator is copied and will not
        be modified at the parent process.
        """
        print("EVALUATE PARALLEL")
        max_proc = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_proc)
        
        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator, 
                    args=[self._species[i]._members[j]]+list(args), 
                    kwds=kwargs
                )

        for key in results:
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key].get())

        pool.close()
        pool.join()
        self.evolve()

    def get_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """Return the true (calculated) population size."""
        return sum([len(s._members) for s in self._species])

    def get_current(self):
        """Get the current genome for evaluation."""
        s = self._species[self._current_species]
        return s._members[self._current_genome]

    def get_current_species(self):
        """Get index of current species being evaluated."""
        return self._current_species

    def get_current_genome(self):
        """Get index of current genome being evaluated."""
        return self._current_genome

    def get_generation(self):
        """Get the current generation number of this population."""
        return self._generation

    def get_species(self):
        """Get the list of species and their respective member genomes."""
        return self._species

    def save(self, filename):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)
