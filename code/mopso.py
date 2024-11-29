import numpy as np
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


# Particle class
class Particle:
    # Initialize particle with random position and velocity
    def __init__(self):
        self.position = np.random.rand(num_dimensions)
        self.clip_position()
        self.velocity = np.random.rand(num_dimensions)
        self.best_position = np.copy(self.position)
        self.best_cost = None

    # Clip particle values
    def clip_position(self):
        for i in range(self.position):
            # Clip continuous parameters
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

            # Round values for discrete parameters (criterion and optimizer)
            if i >= len(continuous_bounds):
                self.position[i] = round(self.position[i])
    
    # Evaluate particle fitness
    def evaluate_fitness(self):
        # Create model with given particle parameters
        model = ModelHandler(list(self.position))

        # Train model and run model to get error and complexity
        model.train(train_loader=None) # TODO Change this to the actual train loader
        error = 1 - model.evaluate(test_loader=None) # TODO Change this to the actual test loader
        complexity = model.count_parameters()

        # Return fitness score
        return error, complexity
    
    # Update particle velocity
    def update_velocity(self, global_best_position):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_velocity = c1 * r1 * (self.best_position - self.position) # Influence of particle memory
        social_velocity = c2 * r2 * (global_best_position - self.position) # Influence of the swarm
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
    
    # Update particle position
    def update_position(self):
        self.position += self.velocity
        self.clip_position()



# Returns TRUE if cost1 dominates cost2 (i.e., cost1 is better than cost2)
def dominates(cost1, cost2):
    return all(x <= y for x, y in zip(cost1, cost2)) and any(x < y for x, y in zip(cost1, cost2))


# PSO loop
def run_pso():
    # Initialize swarm
    swarm = [Particle() for _ in range(num_particles)]
    pareto_front = []
    global_best_position = None

    # Main PSO loop
    for _ in range(num_iterations):
        for particle in swarm:
            # Evaluate particle fitness
            costs = particle.evaluate_fitness()

            # Update particle's personal best
            if particle.best_cost is None or dominates(costs, particle.best_cost):
                particle.best_position = np.copy(particle.position)
                particle.best_cost = costs
            
            # Update Pareto front
            if not any(dominates(other_costs, costs) for _, other_costs in pareto_front):
                pareto_front.append((np.copy(particle.position), costs))
                pareto_front = [entry for entry in pareto_front if not dominates(costs, entry[1])]

        global_best_position = np.random.choice([pos for pos, _ in pareto_front]) # TODO Adjust later

        # Update particle velocities and positions
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()

    # Return best parameters
    print("Best parameters")
    for dim, val in zip(dimensions, global_best_position):
        print(f"   {dim}: {val}")
    return global_best_position
