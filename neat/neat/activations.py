import math

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