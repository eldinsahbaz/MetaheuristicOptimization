import numpy as np
from multiprocessing import Pool
from functools import partial
from pprint import pprint


def dominates(x, y):
    val = np.all(x <= y) and np.any(x < y)

    return val


class best(object):
    def __init__(self, num_variables):
        self.X = np.zeros(num_variables)
        self.O = np.inf


class particle(object):
    def __init__(self, upper_bound, lower_bound, num_variables):
        self.X = (upper_bound-lower_bound) * np.random.random(num_variables) + lower_bound
        self.V = np.zeros(num_variables)
        self.O = np.inf

        self.personal_best = best(num_variables)


class swarm(object):
    def __init__(self, upper_bound, lower_bound, num_variables, num_particles):
        self.Particles = [particle(upper_bound, lower_bound, num_variables) for i in range(num_particles)]
        self.global_best = best(num_variables)


def calculate_objective_vals(objective_function, Particle):
    currentX = Particle.X
    Particle.O = objective_function(currentX)

    # Update the personal_best
    if Particle.O < Particle.personal_best.O:
        Particle.personal_best.X = currentX
        Particle.personal_best.O = Particle.O

    return Particle


def compute_new_positions(num_variables, w, c1, c2, lower_bound, upper_bound, max_velocity, min_velocity, global_best, Particle):
    Particle.V = w * Particle.V + c1 * np.random.random(num_variables) * (Particle.personal_best.X - Particle.X) + c2 * np.random.random(num_variables) * (global_best.X - Particle.X)

    # Check velocities and positions
    Particle.V = np.minimum(np.maximum(Particle.V, min_velocity), max_velocity)
    Particle.X = np.minimum(np.maximum(Particle.X + Particle.V, lower_bound), upper_bound)

    return Particle


def PSO(num_variables, lower_bound, upper_bound, objective_function, num_particles, max_iterations, max_w, min_w, c1, c2, max_velocity, min_velocity):
    Swarm = swarm(upper_bound, lower_bound, num_variables, num_particles)

    for t in range(max_iterations):
        # Calcualte the objective value
        pool = Pool()
        Swarm.Particles = pool.map(partial(calculate_objective_vals, objective_function), Swarm.Particles)

        # Update the global_best
        min_particle = sorted(Swarm.Particles, key=lambda x: x.O, reverse=False)[0]

        if min_particle.O < Swarm.global_best.O:
            Swarm.global_best.X = min_particle.X
            Swarm.global_best.O = min_particle.O

        # Update the X and V vectors
        w = max_w - t * ((max_w - min_w) / max_iterations)
        Swarm.Particles = pool.map(partial(compute_new_positions, num_variables, w, c1, c2, lower_bound, upper_bound, max_velocity, min_velocity, Swarm.global_best), Swarm.Particles)

        pool.close()
        pool.join()

        print("Iteration Number: %s, Swarm.global_best.O: %s" % (t, Swarm.global_best.O,))

    return Swarm.global_best.X


# Define the details of the table design problem
num_variables = 100
upper_bounds = np.zeros(num_variables) + 10
lower_bounds = np.zeros(num_variables) - 10
max_velocity = (upper_bounds - lower_bounds) * 0.2
min_velocity = -max_velocity

def objective(x):
    return np.sum(np.square(x))

inputs = {
            'num_variables': num_variables,
            'upper_bound': upper_bounds,
            'lower_bound': lower_bounds,
            'objective_function': objective,
            'num_particles': 500,
            'max_iterations': 500,
            'max_w': 0.9,
            'min_w': 0.2,
            'c1': 2,
            'c2': 2,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity
        }
print(PSO(**inputs))