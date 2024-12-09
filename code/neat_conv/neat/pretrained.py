import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import pickle


from hyperparameters import Hyperparameters
from neat import Brain, ConvBrain
from genome import ConvolutionalGenome, ConvNode, PoolNode
from activations import relu, sigmoid
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import transforms
from torchvision.datasets import ImageNet

from dataloader import DataLoader


def save_best_genome(brain, filename):
    """Save the best genome of the current generation to a file."""
    best_genome = brain.get_fittest()
    with open(filename, 'wb') as f:
        pickle.dump(best_genome, f)
    # print(f"Best genome saved to {filename}")

def load_best_genome(filename):
    """Load the best genome from a file."""
    try:
        with open(filename, 'rb') as f:
            best_genome = pickle.load(f)
        print(f"Best genome loaded from {filename}")
        return best_genome
    except FileNotFoundError:
        print(f"No saved genome found at {filename}")
        return None
    
def save_population(brain, filename):
    """Save the entire population of the current generation to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(brain, f)
    # print(f"Population saved to {filename}")

def load_population(filename):
    """Load the entire population from a file."""
    try:
        with open(filename, 'rb') as f:
            population = pickle.load(f)
        print(f"Population loaded from {filename}")
        return population
    except FileNotFoundError:
        print(f"No saved population found at {filename}")
        return None


class ResNet18WithoutFC(nn.Module):
    def __init__(self, resnet18_model):
        super(ResNet18WithoutFC, self).__init__()
        
        # Remove the fully connected layer and use the feature extractor part
        self.features = nn.Sequential(
            *list(resnet18_model.children())[:-1]
        )

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x 
    
pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)
pretrained.eval()

resnet = ResNet18WithoutFC(pretrained)

def load_image(flattened_image):
    # Reshape the input to separate channels (1024 values for each R, G, B)
    R = flattened_image[:1024].reshape(32, 32)  # Red channel
    G = flattened_image[1024:2048].reshape(32, 32)  # Green channel
    B = flattened_image[2048:].reshape(32, 32)  # Blue channel

    # Stack the channels to create a 3x32x32 image
    # Stack the channels and normalize to [0,1] range
    output_image = np.stack((R, G, B), axis=0)
    # Normalize the image to [0, 1] range and convert it to a tensor
    output_image = torch.tensor(output_image, dtype=torch.float32) / 255.0
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    output_image = transform(output_image)

    # output_image = ResNet18_Weights.DEFAULT.transforms()(output_image)
    output_image = output_image.unsqueeze(0)

    return output_image

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    num_correct = 0
    for i in range(len(output)):
        num_correct += (expected[i] == output[i])
    # print(f'Classification Accuracy: {num_correct / len(output)}')
    
    return num_correct / len(output)

def evaluate(genome, images, labels):
    """Evaluates the current genome."""
    predictions = []
    NUM_TO_EVALUATE = 150
    
    for i in range(NUM_TO_EVALUATE):
        loaded = load_image(images[i])
        with torch.no_grad():
            resnet_output = resnet(loaded)
            resnet_output = torch.flatten(resnet_output)

            cifar_output = genome.forward(list(resnet_output.squeeze(0)))

            output = []
            for tensor in cifar_output:
                if isinstance(tensor, torch.Tensor):
                    output.append(tensor.item())
                else:
                    output.append(tensor)

            # Convert to float tensor before applying softmax
            output_tensor = torch.tensor(output, dtype=torch.float32)
            softmax = torch.nn.Softmax(dim=0)
            prob = softmax(output_tensor)

            predicted_class = torch.argmax(prob).item()

            top_k_probs, top_k_indices = torch.topk(prob, 3)
            if labels[i] in top_k_indices.tolist():
                predicted_class = labels[i]

            predictions.append(predicted_class)
    
    accuracy = fitness(labels[:NUM_TO_EVALUATE], predictions)
    return float(accuracy)  # Ensure we return a Python float

def main():
    cifar = DataLoader('cifar')
    sample_cifar_batch = cifar.data[0]
    sample_image = sample_cifar_batch[0]
    NUM_IN_BATCH = sample_cifar_batch.shape[0]
    labels = cifar.labels[0]
    sample_label = labels[0]



    # print(f'Accuracy: {(num_correct / num_total) * 100}%')

    nodes_list = [0.01, 0.05, 0.1, 0.09, 0.01, 0.15, 0.2, 0.1]
    edges_list = [0.09, 0.05, 0.1, 0.01, 0.09, 0.15, 0.1, 0.15]
    weight_perturbs_list = [0.4, 0.4, 0.3, 0.4, 0.3, 0.2, 0.2, 0.25]
    weight_sets_list = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1] 

    for i in range(len(nodes_list)):
        hyperparameters = Hyperparameters()
        hyperparameters.mutation_probabilities['node'] = nodes_list[i]
        hyperparameters.mutation_probabilities['edge'] = edges_list[i]
        hyperparameters.mutation_probabilities['weight_perturb'] = weight_perturbs_list[i]
        hyperparameters.mutation_probabilities['weight_set'] = weight_sets_list[i]
        brain = Brain(inputs=512, 
                    outputs=10, 
                    population=200, 
                    hyperparams=hyperparameters)
        
        brain.generate()
        print(f"Training {i}...")
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
            save_population(brain, "cur_population.pkl")
            save_best_genome(brain, "cur_genome.pkl")

  
        best = brain.get_fittest()
        best_accuracy = evaluate(best, sample_cifar_batch, labels[:len(sample_cifar_batch)])
        print(f'Best Accuracy: {best_accuracy}')

        # Save the population and best genome for later use
        save_population(brain, f"exp_population_{i}.pkl")
        save_best_genome(brain, f"exp_best_genome_{i}.pkl")

    # # Print dense layers...
    # edges = best.get_edges()
    # graph = {}
    # for (i, j) in edges:
    #     if not edges[(i, j)].enabled:
    #         continue
    #     if i not in graph:
    #         graph[i] = []
    #     graph[i].append(j)

    # print(f"Best network structure: {best.get_num_nodes()} nodes")
    # for k in sorted(list(graph.keys())):
    #     print(f"{k} - {list(sorted(graph[k]))}")
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


def continue_checkpoint():
    print("Continue checkpoint")
    cifar = DataLoader('cifar')
    sample_cifar_batch = cifar.data[0]
    sample_image = sample_cifar_batch[0]
    NUM_IN_BATCH = sample_cifar_batch.shape[0]
    labels = cifar.labels[0]
    sample_label = labels[0]

    best_genome = load_best_genome('./cur_genome.pkl')
    brain = load_population('./cur_population.pkl')
    print(best_genome)
    
    # brain._hyperparams.mutation_probabilities = {
    #         'node' : 0.1,
    #         'edge' : 0.1,   # previously 0.09
    #         'weight_perturb' : 0.4,
    #         'weight_set' : 0.1,
    #         'bias_perturb' : 0.2,
    #         'bias_set' : 0.1,

    #         'conv_add_node': 0.5,
    #         'conv_delete_node': 0.0,
    #         'conv_kernel': 0.5
    #     }
    # brain._hyperparams.max_generations += 30


    # while brain.should_evolve():
    #     brain.evaluate_parallel(evaluate, images=sample_cifar_batch, labels=labels[:len(sample_cifar_batch)])
    #     # Print training progress
    #     current_gen = brain.get_generation()
    #     current_best = brain.get_fittest()
    #     print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
    #         current_best.get_fitness() * 100, 
    #         current_gen, 
    #         brain._hyperparams.max_generations
    #     ))
    #     save_population(brain, "cur_population.pkl")
    #     save_best_genome(brain, "cur_genome.pkl")

    # best_genome = brain.get_fittest()
    num_total, num_correct = 0, 0
    for i in range(500):
        sample_image, sample_label = sample_cifar_batch[i], labels[i]
        loaded = load_image(sample_image)

        resnet_output = resnet(loaded)
        resnet_output = torch.flatten(resnet_output)

        cifar_output = best_genome.forward(list(resnet_output.squeeze(0)))

        output = []
        for tensor in cifar_output:
            if isinstance(tensor, torch.Tensor):
                output.append(tensor.item())
            else:
                output.append(tensor)

        # Convert to float tensor before applying softmax
        output_tensor = torch.tensor(output, dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=0)
        prob = softmax(output_tensor)

        # predicted_class = torch.argmax(prob).item()
        # if predicted_class == sample_label:
        #     num_correct += 1

        top_k_probs, top_k_indices = torch.topk(prob, 2)
        if sample_label in top_k_indices.tolist():
            num_correct += 1
        
        # print(predicted_class, sample_label)
        num_total += 1
    print(f'Classification Accuracy: {num_correct / num_total}')

    # # Print dense layers...
    # edges = best_genome.get_edges()
    # graph = {}
    # for (i, j) in edges:
    #     if not edges[(i, j)].enabled:
    #         continue
    #     if i not in graph:
    #         graph[i] = []
    #     graph[i].append(j)

    # print(f"Best network structure: {best_genome.get_num_nodes()} nodes")
    # for k in sorted(list(graph.keys())):
    #     print(f"{k} - {list(sorted(graph[k]))}")

if __name__ == "__main__":
    # main()
    continue_checkpoint()