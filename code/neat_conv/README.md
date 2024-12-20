# NEAT-Python

NEAT (NeuroEvolution of Augmenting Topologies) is an algorithm 
developed by Ken Stanley that applies _genetic algorithms_ to machine learning.

1. Generates a population of genomes (neural networks)
2. Clusters genomes into species based on their _genomic distances_
3. Evaluates the _fitness score_ of each genome
4. Breeds and mutates the best genomes over the course of generations

This implementation is a modified version of the algorithm written in Python.

[Here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) is the original paper. Below is an animation of the `flappy_ai.py` demo script.

<!-- <img src="./media/flappy_ai.gif" alt="Flappy AI" width="500"/> -->

## Dependencies

None. Just the standard Python libraries.

## Installation

To install via pip, simply enter `pip install git+https://github.com/SirBob01/NEAT-Python.git` on the console.

## Basic Usage

Import the NEAT module.
```py
from neat import neat
```

Set the hyperparameters of the model. See the source code for the complete list
of tweakable values.
```py
hp = neat.Hyperparameters()
hp.max_generations = 100
hp.distance_weights["bias"] = 0.4
hp.mutation_probabilities["weight_perturb"] = 0.3
```

Generate the genomic population of a new brain, denoting the number of inputs and outputs respectively, as well as its population count, for its base parameters.
```py
# Takes 3 inputs, produces 2 outputs
brain = neat.Brain(3, 2, population=100, hyperparams=hp)
brain.generate()
```

Training genomes can be done in two ways. The first way is via manual iteration:
```py
while brain.should_evolve():
    genome = brain.get_current()
    output = genome.forward([0.3, 0.1, 0.25])

    genome.set_fitness(score(output)) # score() returns a numerical fitness value
    
    brain.next_iteration() # Next genome to be evaluated
```

The second way is to use NEAT-Python's multiprocessing functionality.
```py
def score(genome, some_arg, some_kwarg=None):
    """Calculate the fitness of this genome."""
    output = genome.forward([0.3, 0.1, 0.25])
    example_fitness = sum(output)
    
    print(some_arg, some_kwarg)
    return example_fitness
    
while brain.should_evolve():
    brain.evaluate_parallel(score, 3, some_kwarg="Hello!") # 3, Hello!
```

For both methods, the brain's `.should_evolve()` function determines whether or not to continue evaluating genomes based on the maximum number of generations or fitness score to be achieved.

A genome's `.forward()` function takes a list of input values and produces a list of output values. These outputs may be evaluated by a fitness function and the fitness score of this current genome may be updated via the genome's `.set_fitness()` method.

_Note that the fitness function must be a maximization function, and all values must strictly be non-negative._

To grab a clone of the best performing genome in the population, use the brain's `.get_fittest()` function.

Finally, a brain and all its neural networks can be saved to disk and loaded. Files are automatically read and saved as `.neat` files.
```py
brain.save('filename')
loaded_brain = neat.Brain.load('filename') # Static method
```

Read NEAT's doc-strings for more information on the module's classes and methods.

## TODO
- Implement interspecies sexual crossover
- Fix bugs in repopulation algorithm
- Allow mutable activation functions for each node (heterogeneous activations)

## License

Code and documentation Copyright (c) 2018-2020 Keith Leonardo

Code released under the [BSD 3 License](https://choosealicense.com/licenses/bsd-3-clause/).