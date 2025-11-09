############################################
### Parallelized Optimization Script
### Obtaining Optimal Preference Parameters
############################################

import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from contextlib import contextmanager
import sys, os
import multiprocessing as mp
import dill  # Better serialization for complex objects
import copy

# Set the font family and size to use for Matplotlib figures.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

from road import Road
from junction import Junction
from network import Network

from roads_setup import *
from am_networks_setup import * 
from pm_networks_setup import *

##########################################################################
## Setting up Parameters
##########################################################################

# Simulation time for each network
nt_opt = 600 # Time iterations for simulations in the optimization process
# dt = 0.1 # Time step for the simulation
nt_am_base = 87000 # 1100 - 1325, 2h 25min = 8700 seconds = 87000 time iterations
nt_am_2 = 33600  # 1325 - 1421, 56 min = 3360 seconds = 33600 time iterations
nt_am_3 = 20400 # 1421 - 1455, 34 min = 2040 seconds = 20400 time iterations
nt_pm_base = 18600 # 1455 - 1526, 31 min = 1860 seconds = 18600 time iterations
nt_pm_2 = 37800 # 1526 - 1629, 1h 3min = 3780 seconds = 37800 time iterations
nt_pm_3 = 4200 # 1629 - 1636, 7 min = 420 seconds = 4200 time iterations
nt_pm_4 = 6600 # 1636 - 1647, 11 min = 660 seconds = 6600 time iterations
nt_pm_5 = 13800 # 1647 - 1710, 23 min = 1380 seconds = 13800 time iterations
## Indeed, one can check that the total time is 6 h 10 min, from 1100 - 1710.

# Optimization hyperparameters
grad_size = 1e-3  # For numerically approximating partial derivatives
boundary_tol = 1e-3  # This means that if the descent attempts to escape from the "box", that component will be reset to this value
linesearch_num = 100  # Number of line search iterations (100)
init_step = 1  # Initial step size for gradient descent
decay_factor = 0.5  # Factor by which to decay the step size
max_decay_times = 10  # Maximum number of times to decay the step size
control_factor = 0.5  # Control factor for the Armijo condition
sample_size = 10  # Number of dimensions to sample for gradient estimation
hyperparameters = [grad_size, boundary_tol, linesearch_num, init_step, decay_factor, max_decay_times, control_factor, sample_size]

# Number of processes to use for parallel computation 
num_processes = 10 # If this number is not assigned, we will use all the available CPU cores.
# We recommend that you use num_processes = sample_size as this is the only parallelizable part.

##########################################################################
## Parallelized Gradient Computation Functions
##########################################################################
def run_optimization(network_from, network_to, nt_opt_val, output_dir = None, by_data=False, tag=""):
    Network.prepare_network_densities(network_from, network_to)

    if output_dir:
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
    optimize_network_parallel(network_to, nt_opt_val, hyperparameters, os.path.join(output_dir, "opt_parameters" + tag + ".txt"), by_data = by_data)

def compute_gradient_component(args):
    """
    Worker function to compute a single gradient component.
    
    Args:
        args: Tuple containing (network_data, current_x, base_obj, grad_size, ind, nt, boundary_tol, by_data)
    
    Returns:
        Tuple of (index, gradient_component_value)
    """
    try:
        network_data, current_x, base_obj, grad_size, ind, nt, boundary_tol, by_data = args
        
        # Deserialize the network
        network = dill.loads(network_data)
        
        # Create perturbation vector
        n = len(current_x)
        d = np.zeros(n)
        d[ind] = grad_size
        
        # Compute perturbed objective
        perturbed_x = current_x + d
        perturbed_obj = objective_function(network, perturbed_x, nt, boundary_tol, by_data)
        
        # Compute gradient component
        gradient_component = (perturbed_obj - base_obj) / grad_size
        
        return (ind, gradient_component)
    
    except Exception as e:
        print(f"Error in gradient computation for component {ind}: {e}")
        return (ind, 0.0)  # Return zero gradient in case of error

def objective_function(network, param, nt, boundary_tol, by_data):
    """
    Objective function for optimization.
    
    Args:
        network: Network object
        param: Parameter vector
        nt: Number of time iterations
        boundary_tol: Boundary tolerance
        by_data: Boolean for reset method
    
    Returns:
        Objective function value
    """
    # Set preferences
    network.set_preferences(param)
    max_preference = network.largest_preference()
    min_preference = network.smallest_preference()

    # Check boundary conditions
    boundary_excess = max(max_preference - (1 - boundary_tol), boundary_tol - min_preference)
    if boundary_excess > 0:
        return min(boundary_excess*100*nt, 100*nt)  # Penalty for boundary violation

    # Reset network state
    if by_data:
        network.reset_by_data()
    else:
        network.reset_by_func()

    # Evolve network
    for _ in range(nt):
        network.evolve_resolve()
    
    return -network.get_time_integrated_cars_distance_scaled()

def compute_gradient_parallel(network, current_x, grad_size, sampled_indices, nt, boundary_tol, by_data, num_processes):
    """
    Compute gradient using parallel processing.
    
    Args:
        network: Network object
        current_x: Current parameter vector
        grad_size: Step size for gradient approximation
        sampled_indices: Indices to compute gradient for
        nt: Number of time iterations
        boundary_tol: Boundary tolerance
        by_data: Boolean for reset method
        num_processes: Number of processes to use
    
    Returns:
        Gradient vector
    """
    n = len(current_x)
    G = np.zeros(n)
    
    # Compute base objective with fresh network copy - MINIMAL FIX
    network_copy = dill.loads(dill.dumps(network))
    base_obj = objective_function(network_copy, current_x, nt, boundary_tol, by_data)
    
    # Serialize network for multiprocessing
    network_data = dill.dumps(network)
    
    # Prepare arguments for parallel computation
    args_list = [(network_data, current_x, base_obj, grad_size, ind, nt, boundary_tol, by_data) 
                 for ind in sampled_indices]
    
    # Use multiprocessing to compute gradient components
    with mp.Pool(processes=min(num_processes, len(sampled_indices))) as pool:
        results = pool.map(compute_gradient_component, args_list)
    
    # Collect results
    for ind, gradient_component in results:
        G[ind] = gradient_component
    
    return G

##########################################################################
## Parallelized Optimization Function
##########################################################################

def optimize_network_parallel(network, nt, hyperparameters, save_file_name, by_data=False, num_processes=None):
    """
    Optimize the network using parallelized gradient descent.

    Arguments:
    - network: The network to optimize.
    - nt: Number of iterations for the optimization.
    - hyperparameters: List of hyperparameters for the optimization.
    - save_file_name: Name of the file to save the optimized parameters. Assumes .txt extension.
    - by_data: Default is False. If True, the optimization will be based on stored initial data.
    - num_processes: Number of processes to use. If None, uses all available CPU cores.

    Returns:
    - para: Optimized parameters.
    """
    
    start_time = time.monotonic()
    
    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
        
    # Unpack hyperparameters
    grad_size, boundary_tol, linesearch_num, init_step, decay_factor, max_decay_times, control_factor, sample_size = hyperparameters
    print(f"Using {min(num_processes, sample_size)} processes for parallel gradient computation")
    n = network.get_dim()  # Number of parameters to optimize

    # Set the initial parameters for the network
    para = np.array(network.get_default_preferences())
    network.set_preferences(para)
    #network.resolve()  # Initialize by resolving

    # Initialize the loss storage
    loss_sto = []

    # Gradient descent loop
    for i in range(1, linesearch_num + 1):
        iteration_start_time = time.monotonic()
        print(f"Iteration Number: {i} Out of {linesearch_num}")

        para_temp = para
        # Use fresh network copy for base objective - MINIMAL FIX
        network_obj = dill.loads(dill.dumps(network))
        base_obj = objective_function(network_obj, para_temp, nt, boundary_tol, by_data)
        loss_sto.append(base_obj)
        
        # Sample indices for gradient computation
        sampled_indices = np.random.choice(n, sample_size, replace=False)
        
        # Compute gradient using parallel processing
        G = compute_gradient_parallel(network, para_temp, grad_size, sampled_indices, 
                                    nt, boundary_tol, by_data, num_processes)
        
        G_norm = np.linalg.norm(G)
        if G_norm < 1e-8:
            continue  # Skip this iteration if the gradient is too small to avoid division by zero
        G_normed = G / G_norm

        # Line search
        step = init_step
        n_decay = 0

        while True:
            # Use fresh network copy for line search - MINIMAL FIX
            network_ls = dill.loads(dill.dumps(network))
            ls_obj = objective_function(network_ls, para_temp - step * G_normed, nt, boundary_tol, by_data)
            if ls_obj <= base_obj - control_factor * step * G_norm or n_decay >= max_decay_times:
                break
            step *= decay_factor
            n_decay += 1

        if n_decay < max_decay_times:
            pre_descent_para = para_temp - step * G_normed
        else:
            pre_descent_para = para_temp

        para = pre_descent_para
        
        iteration_end_time = time.monotonic()
        iteration_elapsed = iteration_end_time - iteration_start_time
        print(f"Iteration {i} completed in {iteration_elapsed:.2f} seconds")

    print("Optimization completed!")
    print("Optimal Value of parameters =", para)

    end_time = time.monotonic()
    print(f"\nComputational Time Elapsed: {end_time - start_time:.2f} seconds")

    # Save the optimized parameters to a file
    with open(save_file_name, 'w') as f:
        for p in para:
            f.write(f"{p}\n")
    
    # Save the loss plot
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.xlabel('Iteration')
    plt.ylabel('Weighted Time-Integrated Cars')
    plt.grid()
    plt.plot([i for i in range(1, linesearch_num + 1)], -np.array(loss_sto), color='C0', linestyle='-', linewidth=2)
    plt.title('Weighted Time Integrated Cars on Main Roads (Parallelized)')
    plt.savefig(save_file_name.replace('.txt', '_loss_parallel.png'))
    plt.close(fig)

    return para

##########################################################################
## Original Functions (kept for compatibility)
##########################################################################

def optimize_network(network, nt, hyperparameters, save_file_name, by_data=False):
    """
    Original optimization function (sequential).
    """
    start_time = time.monotonic()
    grad_size, boundary_tol, linesearch_num, init_step, decay_factor, max_decay_times, control_factor, sample_size = hyperparameters
    n = network.get_dim()

    def gradient(x, l, sampled_indices):
        current_x = copy.deepcopy(x)
        G = np.zeros(n)
        base_obj = obj(current_x)
        for ind in sampled_indices:
            d = np.zeros(n)
            d[ind] = l
            G[ind] = (obj(current_x + d) - base_obj) / l
        return G

    para = np.array(network.get_default_preferences())
    network.set_preferences(para)

    if by_data:
        def obj(param):
            network.set_preferences(param)
            max_preference = network.largest_preference()
            min_preference = network.smallest_preference()
            boundary_excess = max(max_preference - (1 - boundary_tol), boundary_tol - min_preference)
            if boundary_excess > 0:
                return min(boundary_excess * 100 * nt, 100 * nt)
            network.reset_by_data()
            for _ in range(nt):
                network.evolve_resolve()
            return -network.get_time_integrated_cars_distance_scaled()
    else:
        def obj(param):
            network.set_preferences(param)
            max_preference = network.largest_preference()
            min_preference = network.smallest_preference()
            boundary_excess = max(max_preference - (1 - boundary_tol), boundary_tol - min_preference)
            if boundary_excess > 0:
                return min(boundary_excess * 100 * nt, 100 * nt)
            network.reset_by_func()
            for _ in range(nt):
                network.evolve_resolve()
            return -network.get_time_integrated_cars_distance_scaled()

    loss_sto = []

    for i in range(1, linesearch_num + 1):
        iteration_start_time = time.monotonic()
        print("Iteration Number:", i, "Out of", linesearch_num)

        para_temp = para
        base_obj = obj(para_temp)
        loss_sto.append(base_obj)
        G = gradient(para_temp, grad_size, sampled_indices=np.random.choice(n, sample_size, replace=False))
        G_norm = np.linalg.norm(G)
        if G_norm < 1e-8:
            continue  # Skip this iteration if the gradient is too small to avoid division by zero
        G_normed = G / G_norm

        step = init_step
        n_decay = 0

        while obj(para_temp - step * G_normed) > base_obj - control_factor * step * G_norm and n_decay < max_decay_times:
            step *= decay_factor
            n_decay += 1

        if n_decay < max_decay_times:
            pre_descent_para = para_temp - step * G_normed
        else:
            pre_descent_para = para_temp

        para = pre_descent_para
        
        iteration_end_time = time.monotonic()
        iteration_elapsed = iteration_end_time - iteration_start_time
        print(f"Iteration {i} completed in {iteration_elapsed:.2f} seconds")

    print("That's it!")
    print("Optimal Value of parameters =", para)

    end_time = time.monotonic()
    print("\nComputational Time Elapsed:", end_time - start_time, "seconds")

    with open(save_file_name, 'w') as f:
        for p in para:
            f.write(f"{p}\n")
    
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.xlabel('Iteration')
    plt.ylabel('Weighted Time-Integrated Cars')
    plt.grid()
    plt.plot([i for i in range(1, linesearch_num + 1)], -np.array(loss_sto), color='C0', linestyle='-', linewidth=2)
    plt.title('Weighted Time Integrated Cars on Main Roads')
    plt.savefig(save_file_name.replace('.txt', '_loss.png'))
    plt.close(fig)

    return para

def evolve_and_initialize_for_next_network(network_from, network_to, nt, para, by_data=False):
    """
    Evolve the network for a given number of iterations and initialize the data for the next network.
    """
    start_time = time.monotonic()

    if by_data:
        network_from.reset_by_data()
    else:
        network_from.reset_by_func()
    
    network_from.set_preferences(para)

    for _ in range(nt):
        network_from.evolve_resolve()

    road_list_from = network_from.get_roads()
    road_list_to = network_to.get_roads()
    
    name_to_to_road_dict = {}
    for r_to in road_list_to:
        name_to_to_road_dict[r_to.get_name()] = r_to

    from_road_to_to_road_dict = {}
    for r_from in road_list_from:
        r_to = name_to_to_road_dict.get(r_from.get_name())
        if r_to is not None:
            from_road_to_to_road_dict[r_from] = r_to

    for r_from, r_to in from_road_to_to_road_dict.items():
        r_to.set_initial_density(r_from.get_current_density())

    print("\nEvolution for current network is completed.")
    end_time = time.monotonic()
    print(f"\nComputational Time Elapsed: {end_time - start_time:.2f} seconds")

def evolve_network(network, nt, para, by_data=False):
    """
    Evolve the network for a given number of iterations.
    """
    start_time = time.monotonic()
    if by_data:
        network.reset_by_data()
    else:
        network.reset_by_func()
    
    network.set_preferences(para)

    for _ in range(nt):
        network.evolve_resolve()

    print("\nEvolution for current network is completed.")
    end_time = time.monotonic()
    print(f"\nComputational Time Elapsed: {end_time - start_time:.2f} seconds")

##########################################################################
## Main Optimization Loop (Parallelized Version)
##########################################################################

if __name__ == "__main__":
    network_am_base, network_am_2, network_am_3 = setup_am_networks(0.5)
    network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5 = setup_pm_networks(0.5)
    # Use the parallelized optimization function
    print("Starting parallelized optimization...")
    
    para_am_base = optimize_network_parallel(network_am_base, nt_opt, hyperparameters, "am_base_parallel.txt", by_data=False, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_am_base, network_am_2, nt_am_base, para_am_base, by_data=False)

    print("AM_base is done!")

    para_am_2 = optimize_network_parallel(network_am_2, nt_opt, hyperparameters, "am_2_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_am_2, network_am_3, nt_am_2, para_am_2, by_data=True)

    print("AM_2 is done!")

    para_am_3 = optimize_network_parallel(network_am_3, nt_opt, hyperparameters, "am_3_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_am_3, network_pm_base, nt_am_3, para_am_3, by_data=True)

    print("AM_3 is done!")

    para_pm_base = optimize_network_parallel(network_pm_base, nt_opt, hyperparameters, "pm_base_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_pm_base, network_pm_2, nt_pm_base, para_pm_base, by_data=True)

    print("PM_base is done!")

    para_pm_2 = optimize_network_parallel(network_pm_2, nt_opt, hyperparameters, "pm_2_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_pm_2, network_pm_3, nt_pm_2, para_pm_2, by_data=True)

    print("PM_2 is done!")

    para_pm_3 = optimize_network_parallel(network_pm_3, nt_opt, hyperparameters, "pm_3_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_pm_3, network_pm_4, nt_pm_3, para_pm_3, by_data=True)

    print("PM_3 is done!")

    para_pm_4 = optimize_network_parallel(network_pm_4, nt_opt, hyperparameters, "pm_4_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_and_initialize_for_next_network(network_pm_4, network_pm_5, nt_pm_4, para_pm_4, by_data=True)

    print("PM_4 is done!")

    para_pm_5 = optimize_network_parallel(network_pm_5, nt_opt, hyperparameters, "pm_5_parallel.txt", by_data=True, num_processes=num_processes)
    evolve_network(network_pm_5, nt_pm_5, para_pm_5, by_data=True)

    print("PM_5 is done!")
    print("All networks are done!")