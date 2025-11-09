''' 
Toy network consisting of a 2-in-2-out junction.
'''
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, parent_dir) # needed to import from parent directory

from road import Road
from junction import Junction
from network import Network
import optimization_script_parallel

import matplotlib.pyplot as plt
import numpy as np

# Set global parameters for the roads
Road.dt = 0.1 # time step in seconds
Road.p_j = 200 # jam density in vehicles per mile

# Create 4 roads in the network
road1 = Road('road1')
road2 = Road('road2')
road3 = Road('road3')
road4 = Road('road4')

# Set parameters for the roads
road1.set_params(speed=20,lanes=1,length=0.5,cap=500,init_density_factor=.5)
road2.set_params(speed=20,lanes=1,length=0.5,cap=500,init_density_factor=.7)

road1.set_left_boundary_function(lambda time,sig: 0.5) # constant inflow of 0.5 vehicles per second
road2.set_left_boundary_function(lambda time,sig: 0.7) # constant inflow of 0.7 vehicles per second

road3.set_params(speed=20,lanes=1,length=0.5,cap=500,init_density_factor=0)
road4.set_params(speed=20,lanes=1,length=0.5,cap=500,init_density_factor=0)

road3.set_right_boundary_function(lambda time,sig: 0) # free outflow
road4.set_right_boundary_function(lambda time,sig: 0) # free outflow

# Create a junction connecting the roads
junction = Junction('2-in-2-out')
junction.set_roads_in(road1, road2).set_roads_out(road3, road4)

# Mark downstream roads as exit links so they evolve each step
junction.set_is_exit_junction()

# Create a network with the junction
network = Network([junction])

# Set driving preferences for the network
network.set_preferences(network.get_default_preferences())

# Optional: if using optimization to set preferences, wrap in a main guard to avoid issues with multiprocessing
if __name__ == "__main__":

    # Optional: optimize the preferences, but not necessary for this toy example
    # # Set the exit junction for the optimization
    # network.compute_distances(junction)
    
    # optimization_script_parallel.hyperparameters[-1] = 1  # number of samples for optimization, need to pick 1 here for toy example
    # optimization_script_parallel.run_optimization(None, network, nt_opt_val=10, output_dir="toy")
    # # Optimal parameters will be saved in output_dir/opt_parameters.txt
    # network.read_preferences(os.path.join("toy", "opt_parameters.txt"))

    # Plot the road's initial densities
    road1.plot_current_density()
    road2.plot_current_density()
    road3.plot_current_density()
    road4.plot_current_density()

    # Simulate the network for 1000 time steps
    for _ in range(1000):
        network.evolve_resolve(record_densities=True)

    # Animate the road's densities over time
    road1.animate_density_history(os.path.join("toy", "road1_animation.gif"))
    road2.animate_density_history(os.path.join("toy", "road2_animation.gif"))
    road3.animate_density_history(os.path.join("toy", "road3_animation.gif"))
    road4.animate_density_history(os.path.join("toy", "road4_animation.gif"))