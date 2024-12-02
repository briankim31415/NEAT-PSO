import math
import numpy as np

def sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))

def tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)

def LReLU(x):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x
    
def relu(x):
    """ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0
    
def softmax(x):
    # Convert the input list to a numpy array for easier handling
    x = np.array(x)
    
    # Subtract the max value for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x))
    
    # Return the normalized probabilities (softmax)
    return exp_x / np.sum(exp_x)

def hello():
    print('hello')