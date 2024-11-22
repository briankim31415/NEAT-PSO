'''
WORK IN PROGRESS
'''



def pareto_dominates(solution_a, solution_b):
    """
    Returns True if solution_a dominates solution_b.
    Solution is a tuple (accuracy, complexity).
    """
    accuracy_a, complexity_a = solution_a
    accuracy_b, complexity_b = solution_b

    # Solution A dominates solution B if it has better accuracy and/or lower complexity
    return accuracy_a <= accuracy_b and complexity_a <= complexity_b and (accuracy_a < accuracy_b or complexity_a < complexity_b)

def update_pareto_front(pareto_front, new_solution):
    """
    Updates the Pareto front with a new solution by adding or replacing it.
    """
    to_remove = []

    # Compare the new solution against the current Pareto front
    for i, solution in enumerate(pareto_front):
        if pareto_dominates(new_solution, solution):
            to_remove.append(i)
        elif pareto_dominates(solution, new_solution):
            return  # New solution is dominated, don't add it

    # Remove dominated solutions
    for i in sorted(to_remove, reverse=True):
        del pareto_front[i]

    # Add the new solution to the front
    pareto_front.append(new_solution)

# Initialize the Pareto front
pareto_front = []

# Example: Evaluate particle solutions
for particle in particles:
    accuracy, complexity = fitness_function(particle)
    new_solution = (accuracy, complexity)
    update_pareto_front(pareto_front, new_solution)

# Pareto front contains the non-dominated solutions
print(f"Pareto front: {pareto_front}")








def update_particles(particles, pareto_front):
    # Update particles' positions and velocities here
    for particle in particles:
        # Update position based on velocity, etc.
        # After updating, calculate fitness
        accuracy, complexity = fitness_function(particle)
        new_solution = (accuracy, complexity)

        # Update Pareto front with the new solution
        update_pareto_front(pareto_front, new_solution)

# Example main PSO loop
for iteration in range(num_iterations):
    update_particles(particles, pareto_front)

# After PSO is done, pareto_front contains all non-dominated solutions