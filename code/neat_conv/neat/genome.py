import random 
import itertools
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from activations import *

from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights

from collections import OrderedDict

class PretrainedLayerBank:
    """Bank of pretrained convolutional and pooling layers from popular architectures"""
    def __init__(self):
        self.conv_layers = []
        self.pool_layers = []
        
        # Load pretrained models
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract layers from VGG16
        for layer in vgg.features:
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.detach().cpu().numpy()
                self.conv_layers.append({
                    'kernel_size': layer.kernel_size[0],
                    'stride': layer.stride[0],
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'weights': weights
                })
            elif isinstance(layer, nn.MaxPool2d):
                self.pool_layers.append({
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'operation': 'max'
                })

        
        # Extract layers from ResNet18
        def extract_conv_layers(module):
            for layer in module.children():
                if isinstance(layer, nn.Conv2d):
                    weights = layer.weight.detach().cpu().numpy()
                    self.conv_layers.append({
                        'kernel_size': layer.kernel_size[0],
                        'stride': layer.stride[0],
                        'in_channels': layer.in_channels,
                        'out_channels': layer.out_channels,
                        'weights': weights
                    })
                elif isinstance(layer, nn.MaxPool2d):
                    self.pool_layers.append({
                        'kernel_size': layer.kernel_size,
                        'stride': layer.stride,
                        'operation': 'max'
                    })
                elif hasattr(layer, 'children'):
                    extract_conv_layers(layer)
                    
        extract_conv_layers(resnet)
        
    def get_random_conv_layer(self, desired_in_channels):
        """Get a random pretrained conv layer, adjusting input channels if needed"""
        layer_info = random.choice(self.conv_layers)
        weights = layer_info['weights']
        
        # Adjust input channels if needed while preserving pretrained weights
        if layer_info['in_channels'] != desired_in_channels:
            old_weights = weights
            new_weights = np.zeros((
                layer_info['out_channels'],
                desired_in_channels,
                layer_info['kernel_size'],
                layer_info['kernel_size']
            ))
            
            # Copy weights for existing channels, initialize rest randomly
            channels_to_copy = min(desired_in_channels, layer_info['in_channels'])
            new_weights[:, :channels_to_copy] = old_weights[:, :channels_to_copy]
            
            if desired_in_channels > layer_info['in_channels']:
                # Initialize remaining channels using Kaiming initialization
                fan_in = desired_in_channels * layer_info['kernel_size'] * layer_info['kernel_size']
                std = math.sqrt(2.0 / fan_in)
                new_weights[:, channels_to_copy:] = np.random.normal(0, std, 
                    (layer_info['out_channels'], desired_in_channels - channels_to_copy, 
                     layer_info['kernel_size'], layer_info['kernel_size']))
            
            weights = new_weights
            
        return ConvNode(
            kernel_size=layer_info['kernel_size'],
            stride=layer_info['stride'],
            activation='relu',
            filter_weights=weights
        )
    
    def get_random_pool_layer(self):
        """Get a random pooling layer configuration"""
        layer_info = random.choice(self.pool_layers)
        return PoolNode(
            kernel_size=layer_info['kernel_size'],
            stride=layer_info['stride'],
            pool_operation=layer_info['operation'],
            activation='relu'
        )


class Edge(object):
    """A gene object representing an edge in the neural network."""
    def __init__(self, weight):
        self.weight = weight
        self.enabled = True

class Node(object):
    """A gene object representing a node in the neural network."""
    def __init__(self, activation):
        self.output = 0
        self.bias = 0
        self.activation = activation

class ConvNode(object):
    """Gene object representing a convolutional layer """
    def __init__(self, kernel_size, stride, activation, filter_weights) -> None: 
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.filter_weights = filter_weights  # np array to do convolution, most likely

class PoolNode(object):
    """Gene object representing a pooling layer"""
    def __init__(self, kernel_size, stride, pool_operation, activation) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_operation = pool_operation
        self.activation = activation

        
def conv_genomic_distance(conv_genome_a, conv_genome_b, distance_weights, conv_weights):
    conv_distance = 0
    dense_distance = genomic_distance(conv_genome_a.dense_layers, conv_genome_b.dense_layers, distance_weights)

    # Compare convolutional architectures
    num_nodes = max(len(conv_genome_a._conv_nodes), len(conv_genome_b._conv_nodes))
    if num_nodes == 0:
        return dense_distance

    # Calculate structural differences
    len_diff = abs(len(conv_genome_a._conv_nodes) - len(conv_genome_b._conv_nodes))
    structural_distance = conv_weights['structure_term'] * len_diff / num_nodes

    # Compare matching nodes
    weight_diff = 0
    kernel_diff = 0
    stride_diff = 0
    matching_nodes = 0

    min_conv_nodes = min(len(conv_genome_a._conv_nodes), len(conv_genome_b._conv_nodes))
    for conv_node_index in range(min_conv_nodes):
        node_a = conv_genome_a._conv_nodes[conv_node_index]
        node_b = conv_genome_b._conv_nodes[conv_node_index]
        
        # Skip if nodes are not of the same type (conv vs pool)
        if type(node_a) != type(node_b):
            continue
        
        if isinstance(node_a, ConvNode) or isinstance(node_a, PoolNode):
            if isinstance(node_a, ConvNode):
                # Compare filter weights if shapes match
                if node_a.filter_weights.shape == node_b.filter_weights.shape:
                    weight_diff += np.mean(np.abs(node_a.filter_weights - node_b.filter_weights))
            
            # Compare kernel sizes and strides for both Conv and Pool nodes
            kernel_diff += abs(node_a.kernel_size - node_b.kernel_size)
            stride_diff += abs(node_a.stride - node_b.stride)

        matching_nodes += 1

    if matching_nodes > 0:
        # Normalize differences
        weight_term = conv_weights['weight_term'] * weight_diff / matching_nodes
        kernel_term = conv_weights['kernel_term'] * kernel_diff / matching_nodes
        stride_term = conv_weights['stride_term'] * stride_diff / matching_nodes
        conv_distance = structural_distance + weight_term + kernel_term + stride_term

    print(f"Convolutional Distance: {conv_distance}, Dense Distance: {dense_distance}")
    return conv_distance + dense_distance


def genomic_distance(a, b, distance_weights):
    """Calculate the genomic distance between two genomes."""
    a_edges = set(a._edges)
    b_edges = set(b._edges)

    # Does not distinguish between disjoint and excess
    matching_edges = a_edges & b_edges
    disjoint_edges = (a_edges - b_edges) | (b_edges - a_edges)
    num_edges = len(max(a_edges, b_edges, key=len))
    num_nodes = min(a._max_node, b._max_node)

    weight_diff = 0
    for edge in matching_edges:
        # edge = (i, j)
        weight_diff += abs(a._edges[edge].weight - b._edges[edge].weight)

    bias_diff = 0
    for node_index in range(num_nodes):
        bias_diff += abs(a._nodes[node_index].bias - b._nodes[node_index].bias)

    t1 = distance_weights['edge'] * len(disjoint_edges) / num_edges
    t2 = distance_weights['weight'] * weight_diff / len(matching_edges)
    t3 = distance_weights['bias'] * bias_diff / num_nodes
    return t1 + t2 + t3

def conv_genomic_crossover(a, b):
    """Breed two convolutional genomes and return the child. 
    Convolutional layers are inherited from the fitter parent, while
    dense layers use standard NEAT crossover.
    """
    child = ConvolutionalGenome(
        a._conv_input_dim, 
        a._conv_output_dim, 
        a._dense_input_dim, 
        a._dense_output_dim, 
        a._conv_default_activation,
        a._dense_default_activation
    )

    # Crossover dense layers using existing NEAT crossover
    child.dense_layers = genomic_crossover(a.dense_layers, b.dense_layers)

    # Inherit convolutional architecture from fitter parent
    fitter_parent = a if a._fitness > b._fitness else b
    less_fit_parent = b if a._fitness > b._fitness else a

    # If equal fitness, randomly choose parent
    if a._fitness == b._fitness:
        fitter_parent, less_fit_parent = random.choice([(a, b), (b, a)])

    child._conv_nodes = []
    min_nodes = min(len(fitter_parent._conv_nodes), len(less_fit_parent._conv_nodes))

    # For matching nodes, randomly inherit from either parent
    for i in range(min_nodes):
        if random.random() < 0.5:
            child._conv_nodes.append(copy.deepcopy(fitter_parent._conv_nodes[i]))
        else:
            child._conv_nodes.append(copy.deepcopy(less_fit_parent._conv_nodes[i]))

    # Inherit remaining nodes from fitter parent
    if len(fitter_parent._conv_nodes) > min_nodes:
        for i in range(min_nodes, len(fitter_parent._conv_nodes)):
            child._conv_nodes.append(copy.deepcopy(fitter_parent._conv_nodes[i]))

    # Ensure channel dimensions match between consecutive layers
    for i in range(len(child._conv_nodes) - 1):
        current_node = child._conv_nodes[i]
        next_node = child._conv_nodes[i + 1]

        if isinstance(current_node, ConvNode) and isinstance(next_node, ConvNode):
            # If channel dimensions don't match, adjust the next node's filter weights
            if current_node.filter_weights.shape[0] != next_node.filter_weights.shape[1]:
                out_channels = next_node.filter_weights.shape[0]
                in_channels = current_node.filter_weights.shape[0]
                kernel_size = next_node.kernel_size
                
                # Create new filter weights with correct dimensions
                next_node.filter_weights = np.random.normal(
                    0, 1, 
                    (out_channels, in_channels, kernel_size, kernel_size)
                )

    return child


def genomic_crossover(a, b):
    """Breed two genomes and return the child. Matching genes
    are inherited randomly, while disjoint genes are inherited
    from the fitter parent.
    """
    # Template genome for child
    child = NeuralNetGenome(a._inputs, a._outputs, a._default_activation)
    a_in = set(a._edges)
    b_in = set(b._edges)

    # Inherit homologous gene from a random parent
    for i in a_in & b_in:
        parent = random.choice([a, b])
        child._edges[i] = copy.deepcopy(parent._edges[i])
    
    # Inherit disjoint/excess genes from fitter parent
    if a._fitness > b._fitness:
        for disjoint_edge in a_in - b_in:
            child._edges[disjoint_edge] = copy.deepcopy(a._edges[disjoint_edge])
    else:
        for disjoint_edge in b_in - a_in:
            child._edges[disjoint_edge] = copy.deepcopy(b._edges[disjoint_edge])
    
    # Calculate max node
    # FInd highest node ID in child's network
    child._max_node = 0
    for (i, j) in child._edges:
        current_max = max(i, j)
        child._max_node = max(child._max_node, current_max)
    child._max_node += 1

    # Inherit nodes
    for n in range(child._max_node):
        inherit_from = []
        if n in a._nodes:
            inherit_from.append(a)
        if n in b._nodes:
            inherit_from.append(b)

        random.shuffle(inherit_from)    # Random choose parent if fitnesses are the same
        # Choose the fitter parent
        parent = max(inherit_from, key=lambda p: p._fitness)
        child._nodes[n] = copy.deepcopy(parent._nodes[n])

    child.reset()
    return child

class BaseGenome:
    """Base Class for standard genome used by the NEAT algorithm."""
    def __init__(self) -> None:
        pass

    def generate(self):
        raise NotImplementedError('generate() not implemented.')

    def forward(self, inputs):
        raise NotImplementedError('forward() not implemented.')
    
    def mutate(self, probabilities):
        raise NotImplementedError('mutate() not implemented')
    
    def is_input(self, n):
        """Determine if the node id is an input."""
        return 0 <= n < self._inputs

    def is_output(self, n):
        """Determine if the node id is an output."""
        return self._inputs <= n < self._unhidden

    def is_hidden(self, n):
        """Determine if the node id is hidden."""
        return self._unhidden <= n < self._max_node

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._edges[i].enabled == False for i in self._edges)

    def get_fitness(self):
        """Return the fitness of the genome."""
        return self._fitness

    def get_nodes(self):
        """Get the nodes of the network."""
        return self._nodes.copy()

    def get_edges(self):
        """Get the network's edges."""
        return self._edges.copy()

    def get_num_nodes(self):
        """Get the number of nodes in the network."""
        return self._max_node

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's internal state."""
        for n in range(self._max_node):
            self._nodes[n].output = 0
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)
    
class ConvolutionalGenome(BaseGenome):
    """Class for a convolutional neural net genome used by the NEAT algorithm."""
    def __init__(self, conv_input_dim, conv_output_dim, dense_input_dim, dense_output_dim, conv_default_activation, dense_default_activation) -> None:
        super().__init__()

        # Nodes
        self._conv_input_dim = conv_input_dim        # node 0   (input)
        self._conv_output_dim = conv_output_dim      # node -1   (output)
        self._conv_default_activation = conv_default_activation

        self._dense_input_dim = dense_input_dim
        self._dense_output_dim = dense_output_dim
        self._dense_default_activation = dense_default_activation

        self._conv_unhidden = 1  # last index of unhidden node (-1, 0)
        self._conv_max_node = 1   # for creating a new_node, start at 1 for convolutional nodes indexing only

        self.layer_bank = PretrainedLayerBank()

        self.choices = [16, 32, 64, 128]
        self.kernel_choices = [2, 3, 4]
        self.stride_choices = [1, 2, 3]
    
        # Structure
        in_channels = 3
        out_channels = self.choices[random.randint(0, len(self.choices) - 1)]
        first_kernel_size = self.kernel_choices[random.randint(0, len(self.kernel_choices) - 1)]
        first_stride = self.stride_choices[random.randint(0, len(self.stride_choices) - 1)]

        second_in_channels = out_channels
        second_out_channels = self.choices[random.randint(0, len(self.choices) - 1)]
        second_kernel_size = self.kernel_choices[random.randint(0, len(self.kernel_choices) - 1)]
        second_stride = self.stride_choices[random.randint(0, len(self.stride_choices) - 1)]

        # Initialize with some pretrained layers
        self._conv_nodes = []
        # # Add 2-3 initial layers from pretrained bank
        # current_channels = 3  # Starting with RGB input
        # num_initial_layers = 3
        # for _ in range(num_initial_layers):
        #     # Add conv layer
        #     conv_node = self.layer_bank.get_random_conv_layer(current_channels)
        #     self._conv_nodes.append(conv_node)
        #     current_channels = conv_node.filter_weights.shape[0]
            
        #     # Maybe add pooling
        #     if random.random() < 0.5:
        #         self._conv_nodes.append(self.layer_bank.get_random_pool_layer())


        self._conv_nodes = [
            ConvNode(
                kernel_size=first_kernel_size,
                stride=first_stride,
                activation='relu',
                filter_weights=np.random.normal(0, 1, (out_channels, in_channels, first_kernel_size, first_kernel_size))
            ),
            PoolNode(
                kernel_size=2,
                stride=2,
                pool_operation='max',
                activation='relu'
            ),
            ConvNode(
                kernel_size=second_kernel_size,
                stride=second_stride,
                activation='relu',
                filter_weights=np.random.normal(0, 1, (second_out_channels, second_in_channels, second_kernel_size, second_kernel_size))
            ),
            PoolNode(
                kernel_size=2,
                stride=2,
                pool_operation='max',
                activation='relu'
            )
        ]
            
        
        self._conv_nodes_to_layers = {}

        self.conv_layers = None
        self.dense_layers = NeuralNetGenome(dense_input_dim, dense_output_dim, dense_default_activation)

        self._fitness = 0
        self._adjusted_fitness = 0

    
    def generate(self):
        # Flow: CIFAR_image_input --> conv_input_dim --> _conv_nodes --> conv_output_dim == _dense_input_dim --> _dense_output_dim
        """Generate a CNN based on the genome configuration."""
        layers = []
        self._conv_nodes_to_layers.clear()

        relu = nn.ReLU(inplace=False)
        height, width, current_channels = 32, 32, 3  # for CIFAR, need to change

        # Create convolutional layers
        for i, conv_node in enumerate(self._conv_nodes):

            if isinstance(conv_node, ConvNode):
                new_conv_layer = nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=conv_node.filter_weights.shape[0],  # Assuming this exists
                        kernel_size=conv_node.kernel_size,
                        stride=conv_node.stride,
                )

                if new_conv_layer.weight.shape != conv_node.filter_weights.shape:
                    raise ValueError(f"Filter weights {new_conv_layer.weight.shape} and {conv_node.filter_weights.shape} are not compatible")

                with torch.no_grad():
                    new_conv_layer.weight.copy_(torch.tensor(conv_node.filter_weights, dtype=torch.float32))

                current_channels = conv_node.filter_weights.shape[0] 
                layers.append(new_conv_layer)
                self._conv_nodes_to_layers[conv_node] = new_conv_layer
                layers.append(relu)

            elif isinstance(conv_node, PoolNode):
                if conv_node.pool_operation == "max":
                    new_pool_layer = nn.MaxPool2d(
                                        kernel_size=conv_node.kernel_size,
                                        stride=conv_node.stride,)
                    layers.append(new_pool_layer)
                    
                elif conv_node.pool_operation == "avg":
                    new_pool_layer = nn.AvgPool2d(
                                        kernel_size=conv_node.kernel_size,
                                        stride=conv_node.stride,
                                    )
                    layers.append(new_pool_layer)
                self._conv_nodes_to_layers[conv_node] = new_pool_layer
                layers.append(relu)  # Add activation
            
            if i < len(self._conv_nodes) - 1:
                next_node = self._conv_nodes[i + 1]
                if isinstance(next_node, ConvNode):
                    next_node_channels_in = next_node.filter_weights.shape[1]
                    if next_node != current_channels:
                        # Insert 1x1 convolution to match channels if necessary
                        inter_layer = nn.Conv2d(in_channels=current_channels, 
                                                out_channels=next_node_channels_in, 
                                                kernel_size=1, stride=1)
                        layers.append(inter_layer)
                    current_channels = next_node_channels_in  # Update channels after this transformation
        
        # Build the model
        self.conv_layers = layers
        self.dense_layers.generate()

    
    def forward(self, x):
        self.generate()
        
        model = nn.Sequential(*self.conv_layers)
        
        with torch.no_grad():
            layers_to_remove = set()
            # print('Passing through convolutional layers...')

            for i, layer in enumerate(model):
                if i > 0 and isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                    if x.shape[2] < kernel_size[0] or x.shape[3] < kernel_size[1]:
                        # print(f"Skipping layer {layer} due to input size: {x.shape[2:]} being too small for kernel size: {kernel_size}")
                        layers_to_remove.add(layer)
                        continue  # Skip this layer

                x = layer(x)

            # Remove all layers / nodes such that no kernel size is TOO large for an input width/height
            # at any given layer in the model
            # conv_layers_to_remove = self.conv_layers[i+1:]
            self.conv_layers = [layer for layer in self.conv_layers if layer not in layers_to_remove]
            layers_to_nodes = { v:k for k, v in self._conv_nodes_to_layers.items() }
            nodes_to_delete = set()
            for layer_to_remove in layers_to_remove:
                if layer_to_remove in layers_to_nodes:
                    nodes_to_delete.add(layers_to_nodes[layer_to_remove])
            # print(f"REMOVING: {nodes_to_delete}")
            # Remove invalid nodes from genotype
            self._conv_nodes = [node for node in self._conv_nodes if node not in nodes_to_delete]

            global_feature_dim = math.prod(x.shape[1:])
            num_channels = x.shape[1]
            desired_dim = int(math.sqrt(self._conv_output_dim // num_channels))

            adaptive_pooling_layer = nn.AdaptiveAvgPool2d((desired_dim, desired_dim)) 
            x = adaptive_pooling_layer(x)
            x = torch.flatten(x, start_dim=1)

            # print(f'Flattened Shape : {x.shape}')
            current_size = x.shape[1]
            target_size = self._conv_output_dim
            padding_size = target_size - current_size

            if padding_size > 0:
                x = torch.cat([x, torch.ones(1, padding_size)], dim=1)

            x = x.squeeze().tolist()
            
            output = self.dense_layers.forward(x)

            probabilities = softmax(output)
    
        return np.argmax(probabilities)
        
    def mutate(self, probabilities):
        dense_layer_mutation_probs = { k:v for k, v in probabilities.items() if 'conv' not in k }
        conv_layer_mutation_probs = { k:v for k, v in probabilities.items() if 'conv' in k }

        conv_population = list(conv_layer_mutation_probs.keys())
        conv_weights = [probabilities[k] for k in conv_population]
        choice = random.choices(conv_population, weights=conv_weights)[0]

        if choice == 'conv_add_node':
            self.conv_add_node()
        elif choice == 'conv_delete_node':
            self.conv_delete_node()
        elif choice == 'conv_kernel':
            # self.conv_shift_kernel()
            pass

        # Mutate the dense layers
        self.dense_layers.mutate(dense_layer_mutation_probs)

    # def conv_add_node(self):
    #     """Modified to use pretrained layers"""
    #     if len(self._conv_nodes) >= 8:  # Limit maximum depth
    #         return
            
    #     random_index = random.randint(1, len(self._conv_nodes))
        
    #     # Determine input channels from previous conv layer
    #     prev_channels = 3  # default for first layer
    #     for node in reversed(self._conv_nodes[:random_index]):
    #         if isinstance(node, ConvNode):
    #             prev_channels = node.filter_weights.shape[0]
    #             break
        
    #     # Add either conv or pool layer
    #     if random.random() < 0.7:  # Favor conv layers
    #         new_node = self.layer_bank.get_random_conv_layer(prev_channels)
    #         self._conv_nodes.insert(random_index, new_node)
            
    #         # Maybe add pooling after conv
    #         if random.random() < 0.3:
    #             pool_node = self.layer_bank.get_random_pool_layer()
    #             self._conv_nodes.insert(random_index + 1, pool_node)
    #     else:
    #         new_node = self.layer_bank.get_random_pool_layer()
    #         self._conv_nodes.insert(random_index, new_node)

    def conv_add_node(self):
        probability_conv = random.random()
        # Choose random place to insert node
        random_index = random.randint(1, len(self._conv_nodes))
        prev_node = self._conv_nodes[random_index - 1]
        next_node = self._conv_nodes[random_index + 1] if random_index + 1 < len(self._conv_nodes) else None

        new_kernel_size = self.kernel_choices[random.randint(1, len(self.kernel_choices) - 1)]
        new_stride = self.stride_choices[random.randint(0, len(self.stride_choices) - 1)] 

        prev_out_channels = prev_node.filter_weights.shape[0] if isinstance(prev_node, ConvNode) else self.choices[random.randint(0, len(self.choices) - 1)]
        next_in_channels = next_node.filter_weights.shape[1] if isinstance(next_node, ConvNode) else self.choices[random.randint(0, len(self.choices) - 1)]

        if probability_conv >= 0.5:
            new_node = ConvNode(kernel_size=new_kernel_size,
                                stride=new_stride,
                                activation='relu',
                                filter_weights=np.random.normal(0, 1, (next_in_channels, prev_out_channels, new_kernel_size, new_kernel_size)))
        else:
            # add relu node
            pool_operations = ['max', 'avg']
            new_pool_operation = pool_operations[random.randint(0, len(pool_operations) - 1)]
            new_node = PoolNode(kernel_size=new_kernel_size,
                                stride=new_stride,
                                pool_operation=new_pool_operation,
                                activation='relu')

        self._conv_nodes.insert(random_index, new_node)
        self._conv_nodes.insert(random_index + 1, nn.ReLU(inplace=False))

    def conv_delete_node(self):
        # Don't delete input
        # print(f"Deleting node...")
        if len(self._conv_nodes) > 1:
            index_to_delete = random.randint(1, len(self._conv_nodes) - 1)
            self._conv_nodes.pop(index_to_delete)
            # print(f"Deleted {index_to_delete}!")


    def conv_shift_kernel(self):
        # print(f"Mutating convolutional kernel...")
        mutatable_conv_nodes = [node for node in self._conv_nodes[1:] if isinstance(node, ConvNode)]
        if len(mutatable_conv_nodes):
            chosen_conv_node = random.choice(mutatable_conv_nodes)
            filter_shape = chosen_conv_node.filter_weights.shape
            
            # Calculate appropriate scale for the noise
            fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
            std = math.sqrt(2.0 / fan_in) * 0.1  # 10% of the standard initialization
            
            mask = np.random.random(filter_shape) < 0.3  # Only modify 30% of weights
            noise = np.random.normal(0, std, filter_shape)
            chosen_conv_node.filter_weights += mask * noise

    def reset(self):
        """Reset the genome's state."""
        self._fitness = 0
        self._adjusted_fitness = 0
        self.dense_layers.reset()
        # Reset any cached layers
        self.conv_layers = None
        self._conv_nodes_to_layers.clear()


class NeuralNetGenome(BaseGenome):
    """Class for a neural net genome used by the NEAT algorithm."""
    def __init__(self, inputs, outputs, default_activation):
        # print("Constructing NeuralNetGenome")
        # Nodes
        self._inputs = inputs
        self._outputs = outputs

        self._unhidden = inputs+outputs
        self._max_node = inputs+outputs # next node_id tracker (for when creating a new node)

        # Structure
        self._edges = {} # (i, j) : Edge
        self._nodes = {} # NodeID : Node

        self._default_activation = default_activation

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0

    def generate(self):
        """Generate the neural network of this genome with minimal
        initial topology, i.e. (no hidden nodes). Call on genome
        creation.
        """
        # Minimum initial topology, no hidden layer
        for node_index in range(self._max_node):
            # Maps integer node_index to Node object
            self._nodes[node_index] = Node(self._default_activation)

        # Create an edge from each input node to output node
        for input_index in range(self._inputs):
            for output_index in range(self._inputs, self._unhidden):
                self.add_edge(input_index, output_index, random.uniform(-1, 1))
                
    def forward(self, inputs):
        """
        Evaluate inputs and calculate the outputs of the
        neural network via the forward propagation algorithm.
        """
        if len(inputs) != self._inputs:
            raise ValueError("Incorrect number of inputs.")
        

        # Set input values
        for i in range(self._inputs):
            self._nodes[i].output = inputs[i]
        
        # Generate backward-adjacency list 
        _from = {}
        for n in range(self._max_node):
            _from[n] = []

        for (i, j) in self._edges:
            if not self._edges[(i, j)].enabled:
                continue
            _from[j].append(i)

        # Calculate output values for each node
        ordered_nodes = itertools.chain(
            range(self._unhidden, self._max_node),
            range(self._inputs, self._unhidden)
        )
        for j in ordered_nodes:
            ax = 0
            for i in _from[j]:
                ax += self._edges[(i, j)].weight * self._nodes[i].output

            node = self._nodes[j]
            node.output = node.activation(ax + node.bias)
        
        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]

    # override, need implementation for abstract method
    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        probabilities = { k:v for k, v in probabilities.items() if 'conv' not in k }

        if self.is_disabled():
            self.add_enabled()

        population = list(probabilities.keys())
        weights = [probabilities[k] for k in population]
        choice = random.choices(population, weights=weights)[0]

        if choice == "node":
            self.add_node()
        elif choice == "edge":
            (i, j) = self.random_pair()
            self.add_edge(i, j, random.uniform(-1, 1))
        elif choice == "weight_perturb" or choice == "weight_set":
            self.shift_weight(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            self.shift_bias(choice)

        self.reset()

    """Private helper functions for mutate"""
    # private
    def add_node(self):
        """Add a new node between a randomly selected edge,
        disabling the parent edge.
        """
        enabled = [k for k in self._edges if self._edges[k].enabled]
        (i, j) = random.choice(enabled)
        edge = self._edges[(i, j)]
        edge.enabled = False

        new_node = self._max_node
        self._max_node += 1
        self._nodes[new_node] = Node(self._default_activation)

        self.add_edge(i, new_node, 1.0)
        self.add_edge(new_node, j, edge.weight)

    # private
    def add_edge(self, i, j, weight):
        """Add a new connection between existing nodes."""
        if (i, j) in self._edges:
            self._edges[(i, j)].enabled = True
        else:
            self._edges[(i, j)] = Edge(weight)
    
    # private
    def add_enabled(self):
        """Re-enable a random disabled edge."""
        disabled = [e for e in self._edges if not self._edges[e].enabled]

        if len(disabled) > 0:
            self._edges[random.choice(disabled)].enabled = True
        
    # private
    def shift_weight(self, type):
        """Randomly shift, perturb, or set one of the edge weights."""
        e = random.choice(list(self._edges.keys()))
        if type == "weight_perturb":
            self._edges[e].weight += random.uniform(-1, 1)
        elif type == "weight_set":
            self._edges[e].weight = random.uniform(-1, 1)

    # private 
    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming edge."""
        # Select only nodes in the hidden and output layer
        n = random.choice(range(self._inputs, self._max_node))
        if type == "bias_perturb":
            self._nodes[n].bias += random.uniform(-1, 1)
        elif type == "bias_set":
            self._nodes[n].bias = random.uniform(-1, 1)

    # private
    def random_pair(self):
        """Generate random nodes (i, j) such that:
        1. i is not an output
        2. j is not an input
        3. i != j
        """
        i = random.choice([n for n in range(self._max_node) if not self.is_output(n)])
        j_list = [n for n in range(self._max_node) if not self.is_input(n) and n != i]

        if not j_list:
            j = self._max_node
            self.add_node()
        else:
            j = random.choice(j_list)

        return (i, j)
