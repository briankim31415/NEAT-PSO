import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


from hyperparameters import Hyperparameters
from neat import Brain, ConvBrain
from genome import ConvolutionalGenome, ConvNode, PoolNode
from activations import relu, sigmoid

from dataloader import DataLoader

import networkx as nx
import matplotlib.pyplot as plt


def load_image(flattened_image):
    # Reshape the input to separate channels (1024 values for each R, G, B)
    R = flattened_image[:1024].reshape(32, 32)  # Red channel
    G = flattened_image[1024:2048].reshape(32, 32)  # Green channel
    B = flattened_image[2048:].reshape(32, 32)  # Blue channel

    # Stack the channels to create a 3x32x32 image
    # Stack the channels and normalize to [0,1] range
    output_image = np.stack((R, G, B), axis=0) / 255.0
    output_image = torch.tensor(output_image).float().unsqueeze(0)
    return output_image


def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    num_correct = 0
    for i in range(len(output)):
        num_correct += (expected[i] == output[i])
    print(f'Classification Accuracy: {num_correct / len(output)}')
    
    return num_correct / len(output)

def evaluate(genome, images, labels):
    """Evaluates the current genome."""
    predictions = []
    NUM_TO_EVALUATE = 1000
    for i in range(NUM_TO_EVALUATE):
        loaded = load_image(images[i])
        output_label = genome.forward(loaded)
        predictions.append(output_label)
    return fitness(labels[:NUM_TO_EVALUATE], predictions)

def main():
    cifar = DataLoader('cifar')
    sample_cifar_batch = cifar.data[0]
    sample_image = sample_cifar_batch[0]
    NUM_IN_BATCH = sample_cifar_batch.shape[0]
    labels = cifar.labels[0]
    sample_label = labels[0]
   
    hyperparameters = Hyperparameters()
    brain = ConvBrain(conv_inputs=3072,
                      conv_outputs=1028,
                      dense_inputs=1028,
                      dense_outputs=10,
                      population=50,
                      hyperparams=hyperparameters)
    
    brain.generate()
    print("Training...")
    while brain.should_evolve():
        brain.evaluate_parallel(evaluate, images=sample_cifar_batch, labels=labels[:len(sample_cifar_batch)])
        # Print training progress
        current_gen = brain.get_generation()
        current_best = brain.get_fittest()
        print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
            current_best.get_fitness() * 100, 
            current_gen, 
            hyperparameters.max_generations
        ))

    for specie in brain.get_species():
        for individual in specie._members:
            print(individual._conv_nodes)
            individual.generate()
            print(nn.Sequential(*individual.conv_layers))
            print(individual.forward(load_image(sample_image)), sample_label)
            print(f'Fitness: {individual.get_fitness()}')

    best = brain.get_fittest()

    print(nn.Sequential(*best.conv_layers))

    # Print dense layers...
    edges = best.dense_layers.get_edges()
    graph = {}
    for (i, j) in edges:
        if not edges[(i, j)].enabled:
            continue
        if i not in graph:
            graph[i] = []
        graph[i].append(j)

    
    # print(f"Best network structure: {best.dense_layers.get_num_nodes()} nodes")
    # for k in graph:
    #     print(f"{k} - {graph[k]}")

    # G = nx.DiGraph()
    # for i, js in graph.items():
    #     for j in js:
    #         G.add_edge(i, j)

    # # Draw the graph
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G)  # Use spring layout for better visualization
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=12, font_weight='bold')
    # plt.title("Graph Representation of the Best Network Structure")
    # plt.show()

    return


    sample_conv_genome = ConvolutionalGenome(conv_input_dim=3072,
                                             conv_output_dim=128,
                                             dense_input_dim=128,
                                             dense_output_dim=10,
                                             conv_default_activation=relu,
                                             dense_default_activation=relu)
    
    sample_conv_genome.generate()
    for _ in range(20):
        sample_conv_genome.mutate(hyperparameters.mutation_probabilities)
        # x = random.random()
        # if x <= 0.5:
        #     kernel_size = random.randint(3, 5)
        #     stride = random.randint(1, 3)
        #     in_channels, out_channels = random.randint(1, 5), random.randint(1,5)
        #     new_node = ConvNode(kernel_size=kernel_size, stride=stride, activation="relu", filter_weights=np.ones((out_channels, in_channels, kernel_size, kernel_size)))
        # else:
        #     kernel_size = random.randint(3, 5)
        #     stride = random.randint(1, 3)
        #     in_channels, out_channels = random.randint(1, 5), random.randint(1,5)
        #     new_node = PoolNode(kernel_size=kernel_size, stride=stride, pool_operation='max', activation="relu")
            
        # sample_conv_genome._conv_nodes.append(new_node)

        sample_conv_genome.generate()

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