import numpy as np

# Load config file variables
from config_loader import load_config
config = load_config()
num_particles = config['PSO_num_particles']
num_iterations = config['PSO_num_iterations']
w = config['PSO_w']
c1 = config['PSO_c1']
c2 = config['PSO_c2']
dimensions_bounds = config['PSO_dimensions_bounds']

# Extract dimensions and bounds
dimension = [dim for dim in dimensions_bounds]
num_dimensions = len(dimension)
bounds = [b for b in dimensions_bounds.values()]

# Initialize particles
particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_particles, num_dimensions))
velocities = np.zeros_like(particles)
personal_best_positions = np.copy(particles)
personal_best_scores = np.array([fitness_eval(p) for p in particles])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)


def fitness_eval(particle):
    num_conv_layers = particle[0]
    num_filters = particle[1]
    dropout_rate = particle[2]
    # Run epochs of the arch with given position parameters
    # Also involve complexity of the model (less complex is better)
    # Return fitness score
    return 1

def run_pso():
    # PSO Loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            # Update position
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], [b[0] for b in bounds], [b[1] for b in bounds])

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

        # Print progress
        print(f"Iteration {iteration+1}/{num_iterations}, Best Score: {global_best_score}")

    # Return best parameters
    return global_best_position
