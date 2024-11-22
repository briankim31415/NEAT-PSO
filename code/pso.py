import numpy as np
import torch
from model import ModelHandler

# Load config file variables
from config_loader import load_config
config = load_config()
num_particles = config['PSO_num_particles']
num_iterations = config['PSO_num_iterations']
w = config['PSO_w']
c1 = config['PSO_c1']
c2 = config['PSO_c2']
continuous_bounds = config['model_continuous_bounds']
criterion_bounds = [0, len(config['model_criterion_mapping']) - 1]
optimizer_bounds = [0, len(config['model_optimizer_mapping']) - 1]

# Extract dimensions and bounds
dimensions = [dim for dim in continuous_bounds] + ['criterion', 'optimizer']
num_dimensions = len(dimensions)
bounds = [b for b in continuous_bounds.values()] + [criterion_bounds, optimizer_bounds]

# Initialize particles
particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_particles, num_dimensions))
particles = np.array([clip_particle(p) for p in particles])
velocities = np.zeros_like(particles)
personal_best_positions = np.copy(particles)
personal_best_scores = np.array([fitness_eval(p) for p in particles])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)



# Clip particle values
def clip_particle(particle):
    for i in range(len(particle)):
        # Clip continuous parameters
        particle[i] = np.clip(particle[i], bounds[i][0], bounds[i][1])

        # Round values for discrete parameters (criterion and optimizer)
        if i >= len(particle) - 2:
            particle[i] = round(particle[i])
    return particle


# Fitness evaluation function
def fitness_eval(particle):
    # Create model with given particle parameters
    model = ModelHandler(list(particle))

    # Train model and run model to get accuracy
    model.train(train_loader=None) # TODO Change this to the actual train loader
    accuracy = model.evaluate(test_loader=None) # TODO Change this to the actual test loader

    # Get complexity of the model (number of parameters)
    complexity = model.count_parameters()

    # Return fitness score (want to minimize, i.e. less is better)
    return 1 - accuracy


# PSO loop
def run_pso():
    # Go through every particle at each iteration of the PSO loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i] # Influence of the current velocity
                + c1 * r1 * (personal_best_positions[i] - particles[i]) # Influence of the particle memory
                + c2 * r2 * (global_best_position - particles[i]) # Influence of the swarm
            )

            # Update position
            particles[i] += velocities[i]
            particles[i] = clip_particle(particles[i])

            # Evaluate fitness
            fitness = fitness_eval(particles[i])

            # Update personal best
            if fitness < personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = fitness

        # Update global best
        best_particle_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_particle_idx] < global_best_score:
            global_best_position = personal_best_positions[best_particle_idx]
            global_best_score = personal_best_scores[best_particle_idx]

    # Return best parameters
    print("Best parameters")
    for dim, val in zip(dimensions, global_best_position):
        print(f"   {dim}: {val}")
    return global_best_position
