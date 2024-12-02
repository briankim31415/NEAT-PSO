import math
import random
import numpy as np
import torch


from hyperparameters import Hyperparameters
from neat import Brain
from genome import ConvolutionalGenome, ConvNode, PoolNode
from activations import relu, sigmoid

from dataloader import DataLoader

# def fitness(expected, output):
#     """Calculates the similarity score between expected and output."""
#     s = 0
#     for i in range(4):
#         s += (expected[i] - output[i])**2
#     return 1/(1 + math.sqrt(s))

# def evaluate(genome):
#     """Evaluates the current genome."""
#     f1 = genome.forward([0.0, 0.0])[0]
#     f2 = genome.forward([1.0, 0.0])[0]
#     f3 = genome.forward([0.0, 1.0])[0]
#     f4 = genome.forward([1.0, 1.0])[0]
#     return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

def main():

    sample_conv_genome = ConvolutionalGenome(conv_input_dim=3072,
                                             conv_output_dim=128,
                                             dense_input_dim=128,
                                             dense_output_dim=10,
                                             conv_default_activation=relu,
                                             dense_default_activation=relu)
    
    # [output channels, input channels, kernel size, kernel size]
    filter_weights_1 = np.ones((5, 3, 3, 3))
    filter_weights_2 = np.ones((7, 6, 2, 2))

    sample_conv_genome._conv_nodes = [
            ConvNode(kernel_size=3, stride=random.randint(1, 3), activation="relu", filter_weights=filter_weights_1),
            ConvNode(kernel_size=2, stride=random.randint(1, 3), activation="relu", filter_weights=filter_weights_2),
            PoolNode(kernel_size=random.randint(1, 3), stride=random.randint(1, 3), pool_operation='max', activation="relu"),
        ]
    
    
    sample_conv_genome.generate()

    print(sample_conv_genome.conv_layers)
    print(sample_conv_genome.dense_layers)

    cifar = DataLoader('cifar')

    sample_cifar_batch = cifar.data[0]
    sample_image = sample_cifar_batch[0]
    print(sample_image.shape)

    # Reshape the input to separate channels (1024 values for each R, G, B)
    R = sample_image[:1024].reshape(32, 32)  # Red channel
    G = sample_image[1024:2048].reshape(32, 32)  # Green channel
    B = sample_image[2048:].reshape(32, 32)  # Blue channel

    # Stack the channels to create a 3x32x32 image
    output_image = np.stack((R, G, B), axis=0)

    output_image = np.random.rand(1, 3, 32, 32).astype(np.float32) 

    output_image = torch.tensor(output_image)

    print("Running forward...")
    sample_conv_genome.forward(output_image)


    

    # hyperparams = Hyperparameters()
    # hyperparams.max_generations = 300

    # brain = Brain(inputs=2, 
    #               outputs=1, 
    #               population=150, 
    #               hyperparams=hyperparams)
    # brain.generate()
    
    # print("Training...")
    # while brain.should_evolve():
    #     brain.evaluate_parallel(evaluate)

    #     # Print training progress
    #     current_gen = brain.get_generation()
    #     current_best = brain.get_fittest()
    #     print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
    #         current_best.get_fitness() * 100, 
    #         current_gen, 
    #         hyperparams.max_generations
    #     ))

    # best = brain.get_fittest()
    # f1 = best.forward([0.0, 0.0])[0]
    # f2 = best.forward([1.0, 0.0])[0]
    # f3 = best.forward([0.0, 1.0])[0]
    # f4 = best.forward([1.0, 1.0])[0]
    # fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

    # edges = best.get_edges()
    # graph = {}
    # for (i, j) in edges:
    #     if not edges[(i, j)].enabled:
    #         continue
    #     if i not in graph:
    #         graph[i] = []
    #     graph[i].append(j)

    
    # print()
    # print(f"Best network structure: {best.get_num_nodes()} nodes")
    # for k in graph:
    #     print(f"{k} - {graph[k]}")
    # print()
    # print("Accuracy: {:.2f}%".format(fit * 100))
    # print("0 ^ 0 = {:.3f}".format(f1))
    # print("1 ^ 0 = {:.3f}".format(f2))
    # print("0 ^ 1 = {:.3f}".format(f3))
    # print("1 ^ 1 = {:.3f}".format(f4))


if __name__ == "__main__":
    main()