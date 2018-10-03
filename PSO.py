import numpy as np
import scipy.stats as sp
from copy import deepcopy
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
        self.prev_global_best = None


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


def PSO(num_variables, lower_bound, upper_bound, objective_function, num_particles, max_iterations, max_w, min_w, c1, c2, max_velocity, min_velocity, tolerance, patience, disp):
    Swarm = swarm(upper_bound, lower_bound, num_variables, num_particles)
    convergence_curve = list()
    patience_counter = 0

    for t in range(max_iterations):
        pool = Pool()
        Swarm.Particles = pool.map(partial(calculate_objective_vals, objective_function), Swarm.Particles)

        min_particle = sorted(Swarm.Particles, key=lambda x: x.O, reverse=False)[0]

        if Swarm.prev_global_best != None and abs(Swarm.prev_global_best.O - Swarm.global_best.O) <= tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if min_particle.O < Swarm.global_best.O:
            Swarm.prev_global_best = deepcopy(Swarm.global_best)
            Swarm.global_best.X = min_particle.X
            Swarm.global_best.O = min_particle.O

        w = max_w - t * ((max_w - min_w) / max_iterations)
        Swarm.Particles = pool.map(partial(compute_new_positions, num_variables, w, c1, c2, lower_bound, upper_bound, max_velocity, min_velocity, Swarm.global_best), Swarm.Particles)

        pool.close()
        pool.join()

        if disp:
            print("Iteration Number: %s, Global Best: %s" % (t, Swarm.global_best.O,))
        convergence_curve.append(Swarm.global_best.O)

    return (Swarm.global_best.X, convergence_curve)


def visualize_convergence(convergence_curve):
    if np.all(np.array(convergence_curve) > 0):
        plt.yscale("log")

    plt.title("Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.plot(convergence_curve)
    plt.show()

    convergence_curve = list()


def compare_algorithms(best_runs_one, best_runs_two):
    _, pvalue = sp.ranksums(best_runs_one, best_runs_two)

    return pvalue