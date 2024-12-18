import math
import sys, os

from hyperparameters import Hyperparameters
from neat import Brain

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1/(1 + math.sqrt(s))

def evaluate(genome):
    """Evaluates the current genome."""
    f1 = genome.forward([0.0, 0.0])[0]
    f2 = genome.forward([1.0, 0.0])[0]
    f3 = genome.forward([0.0, 1.0])[0]
    f4 = genome.forward([1.0, 1.0])[0]
    return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

def main():
    hyperparams = Hyperparameters()
    hyperparams.max_generations = 300

    brain = Brain(inputs=2, 
                  outputs=1, 
                  population=150, 
                  hyperparams=hyperparams)
    brain.generate()
    
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate)

        # Print training progress
        current_gen = brain.get_generation()
        current_best = brain.get_fittest()
        print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
            current_best.get_fitness() * 100, 
            current_gen, 
            hyperparams.max_generations
        ))

    best = brain.get_fittest()
    f1 = best.forward([0.0, 0.0])[0]
    f2 = best.forward([1.0, 0.0])[0]
    f3 = best.forward([0.0, 1.0])[0]
    f4 = best.forward([1.0, 1.0])[0]
    fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

    edges = best.get_edges()
    graph = {}
    for (i, j) in edges:
        if not edges[(i, j)].enabled:
            continue
        if i not in graph:
            graph[i] = []
        graph[i].append(j)

    
    print()
    print(f"Best network structure: {best.get_num_nodes()} nodes")
    for k in graph:
        print(f"{k} - {graph[k]}")
    print()
    print("Accuracy: {:.2f}%".format(fit * 100))
    print("0 ^ 0 = {:.3f}".format(f1))
    print("1 ^ 0 = {:.3f}".format(f2))
    print("0 ^ 1 = {:.3f}".format(f3))
    print("1 ^ 1 = {:.3f}".format(f4))



if __name__ == "__main__":
    main()