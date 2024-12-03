from activations import *

class Hyperparameters(object):
    """Hyperparameter settings for the Brain object."""
    def __init__(self):
        self.delta_threshold = 1.5
        self.distance_weights = {
            'edge' : 1.0,
            'weight' : 1.0,
            'bias' : 1.0
        }
        self.default_activation = relu

        self.max_fitness = float('inf')
        self.max_generations = 50
        self.max_fitness_history = 30

        self.breed_probabilities = {
            'asexual' : 1.0,    # change temporarily
            'sexual' : 0.0
        }
        self.mutation_probabilities = {
            'node' : 0.01,
            'edge' : 0.09,
            'weight_perturb' : 0.4,
            'weight_set' : 0.1,
            'bias_perturb' : 0.3,
            'bias_set' : 0.1,

            'conv_add_node': 0.15,
            'conv_delete_node': 0.05,
            'conv_kernel': 0.8,
        }