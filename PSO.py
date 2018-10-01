import sys
import numpy as np
from pprint import pprint
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool


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


def robust_expectation_objective(objective, x):
    H, delta = 50, 5e-2
    x = np.array(x)

    original_objective_value = objective(x)
    error_objective_value = np.sum([objective(x + 2 * delta * np.random.random() - delta) for _ in range(H)])
    expectation_objective_value = (original_objective_value + error_objective_value) /  (H + 1)

    return expectation_objective_value


def robust_variace_objective(objective, x):
    H, delta, threshold = 50, 5e-2, 1e-3
    x = np.array(x)

    original_objective_value = objective(x)

    error_objective_value = np.mean([objective(x + 2 * delta * np.random.random() - delta) for _ in range(H)])
    variance_objective_value = abs(error_objective_value - original_objective_value) / abs(original_objective_value)

    if threshold < variance_objective_value:
        return np.inf

    return original_objective_value


def calculate_objective_vals(objective_function, Particle):
    currentX = Particle.X
    Particle.O = objective_function(currentX)

    if Particle.O < Particle.personal_best.O:
        Particle.personal_best.X = currentX
        Particle.personal_best.O = Particle.O

    return Particle


def compute_new_positions(num_variables, w, c1, c2, lower_bound, upper_bound, max_velocity, min_velocity, global_best, Particle):
    Particle.V = w * Particle.V + c1 * np.random.random(num_variables) * (Particle.personal_best.X - Particle.X) + c2 * np.random.random(num_variables) * (global_best.X - Particle.X)

    Particle.V = np.minimum(np.maximum(Particle.V, min_velocity), max_velocity)
    Particle.X = np.minimum(np.maximum(Particle.X + Particle.V, lower_bound), upper_bound)

    return Particle


def PSO(num_variables, lower_bound, upper_bound, objective_function, num_particles, max_iterations, max_w, min_w, c1, c2, max_velocity, min_velocity):
    Swarm = swarm(upper_bound, lower_bound, num_variables, num_particles)

    for t in range(max_iterations):
        pool = Pool()
        Swarm.Particles = pool.map(partial(calculate_objective_vals, objective_function), Swarm.Particles)

        min_particle = sorted(Swarm.Particles, key=lambda x: x.O, reverse=False)[0]

        if min_particle.O < Swarm.global_best.O:
            Swarm.global_best.X = min_particle.X
            Swarm.global_best.O = min_particle.O

        w = max_w - t * ((max_w - min_w) / max_iterations)
        Swarm.Particles = pool.map(partial(compute_new_positions, num_variables, w, c1, c2, lower_bound, upper_bound, max_velocity, min_velocity, Swarm.global_best), Swarm.Particles)

        pool.close()
        pool.join()

        print("Iteration Number: %s, Global Best: %s" % (t, Swarm.global_best.O,))
        convergence_curve.append(Swarm.global_best.O)

    return Swarm.global_best.X


def objective(x):
    i = 0.001
    return -((1/((2*np.pi)**0.5))*np.exp(-0.5*((((x[0]-1.5)*(x[0]-1.5)+(x[1]-1.5)*(x[1]-1.5))/0.5)**1)) + (2/((2*np.pi)**0.5))*np.exp(-0.5*((((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))/i)**1)))



convergence_curve = list()

# Define the details of the table design problem
num_variables = 2
upper_bounds = np.zeros(num_variables) + 10
lower_bounds = np.zeros(num_variables) - 10
max_velocity = (upper_bounds - lower_bounds) * 0.2
min_velocity = -max_velocity

inputs = {
            'num_variables': num_variables,
            'upper_bound': upper_bounds,
            'lower_bound': lower_bounds,
            'objective_function': partial(robust_variace_objective, objective),
            'num_particles': 1000,
            'max_iterations': 50,
            'max_w': 0.9,
            'min_w': 0.2,
            'c1': 2,
            'c2': 2,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity
        }
output = PSO(**inputs)
print(output)

plt.figure()
plt.yscale("log")
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.plot(convergence_curve)
plt.show()