import PSO
import numpy as np
from pprint import pprint
from functools import partial


# Define the details of the table design problem
def objective_one(x):
    i = 0.001
    return -((1/((2*np.pi)**0.5))*np.exp(-0.5*((((x[0]-1.5)*(x[0]-1.5)+(x[1]-1.5)*(x[1]-1.5))/0.5)**1)) + (2/((2*np.pi)**0.5))*np.exp(-0.5*((((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))/i)**1)))


def sphere(x):
      return np.sum(np.square(x))


num_variables = 2
upper_bounds = np.zeros(num_variables) + 10
lower_bounds = np.zeros(num_variables) - 10
max_velocity = (upper_bounds - lower_bounds) * 0.2
min_velocity = -max_velocity



inputs = {
            'num_variables': num_variables,
            'upper_bound': upper_bounds,
            'lower_bound': lower_bounds,
            'objective_function': partial(PSO.robust_variace_objective, objective_one),
            'num_particles': 1000,
            'max_iterations': 10,
            'max_w': 0.9,
            'min_w': 0.2,
            'c1': 2,
            'c2': 2,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity,
            'tolerance': 1e-2,
            'patience': 3,
            'disp': True
        }
best_solns_one = list()
for i in range(10):
    output, convergence_curve = PSO.PSO(**inputs)
    best_solns_one.append(output)


num_variables = 2
upper_bounds = np.zeros(num_variables) + 10
lower_bounds = np.zeros(num_variables) - 10
max_velocity = (upper_bounds - lower_bounds) * 0.2
min_velocity = -max_velocity

inputs = {
            'num_variables': num_variables,
            'upper_bound': upper_bounds,
            'lower_bound': lower_bounds,
            'objective_function': partial(PSO.robust_variace_objective, objective_one),
            'num_particles': 1000,
            'max_iterations': 10,
            'max_w': 0.9,
            'min_w': 0.4,
            'c1': 2,
            'c2': 2,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity,
            'tolerance': 1e-2,
            'patience': 3,
            'disp': True
        }
best_solns_two = list()
for i in range(10):
    output, convergence_curve = PSO.PSO(**inputs)
    best_solns_two.append(output)

print("The difference is significant" if PSO.compare_algorithms(best_solns_one, best_solns_two) < 0.05 else "The difference is not significant")


num_variables = 100
upper_bounds = np.zeros(num_variables) + 10
lower_bounds = np.zeros(num_variables) - 10
max_velocity = (upper_bounds - lower_bounds) * 0.2
min_velocity = -max_velocity

inputs = {
            'num_variables': num_variables,
            'upper_bound': upper_bounds,
            'lower_bound': lower_bounds,
            'objective_function': sphere,
            'num_particles': 1000,
            'max_iterations': 30,
            'max_w': 0.9,
            'min_w': 0.2,
            'c1': 2,
            'c2': 2,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity,
            'tolerance': 1e-2,
            'patience': 3,
            'disp': True
        }


output, convergence_curve = PSO.PSO(**inputs)
pprint(output)
PSO.visualize_convergence(convergence_curve)
