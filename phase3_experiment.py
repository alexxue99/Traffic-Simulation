"""
Gamma + ntopt Parameter Sweep Experiment
"""
import os

from am_networks_setup import setup_am_networks
from pm_networks_setup import setup_pm_networks
import optimization_script_parallel
import numpy as np
import base_script

def run_phase3_experiment(gamma_values, gamma2_values):
    """
    Optimizes all the networks from the base AM network to PM base network, using the inputted gamma values and nt opt set to 10.
    Skips optimizing networks that were already optimized in phase 2.

    Runs the networks using the optimized parameters, and creates animation and pictures.
    
    Stores results to phase3_experiments/.
    """
    
    print("Starting Parameters Sweep")
    print("=" * 50)
    
    for i, gamma_val in enumerate(gamma_values):
        if i < 5:
            continue  # Skip already done experiments
        nt_opt_val = 10  # Fixed nt_opt value for phase 3
        gamma2_val = gamma2_values[i]
        print(f"\nExperiment {i+1}/{len(gamma_values)}: gamma = {gamma_val:.4f}")
        print(f"\nExperiment {i+1}/{len(gamma_values)}: gamma = {gamma2_val:.4f}")
        print("-" * 30)
    
        # Steps 1-2: instead of reloading, just reset everything by constructing using the new gamma
        network_am_base, network_am_2, network_am_3 = setup_am_networks(gamma_val)
        network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5 = setup_pm_networks(gamma_val, gamma2_val)
        
        # Step 3: Create output directory for this gamma
        gamma_str = f"{gamma_val:.4f}"
        gamma2_str = f"{gamma2_val:.4f}"
        output_dir = os.path.join("phase3_experiments", f"gamma_{gamma_str}_gamma2_{gamma2_str}")
        phase2_output_dir = os.path.join("phase2_experiments_trials", f"gamma_{gamma_str}_ntopt_10")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Step 4: Run optimizations with output directory
        print("Starting optimization...")
        networks = (network_am_base, network_am_2, network_am_3,
                    network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5)
        networks_strings = ("AM Base", "AM 2", "AM 3",
                    "PM Base", "PM 2", "PM 3", "PM 4", "PM 5")
        nts = (87000, 33600, 20400, 18600, 37800, 4200, 6600, 13800)
        try:
            prev = None
            for i, network in enumerate(networks):
                new_network = i >= 4

                if new_network:
                    optimization_script_parallel.run_optimization(prev, network,
                                            nt_opt_val, output_dir, tag=f"_{networks_strings[i]}")
                    network.reset_by_data()
                    pref_file = os.path.join(output_dir, "opt_parameters" + f"_{networks_strings[i]}" + ".txt")
                else:
                    print(f"Skipping optimization for network {networks_strings[i]}")
                    pref_file = os.path.join(phase2_output_dir, "opt_parameters" + f"_{networks_strings[i]}_Averaged" + ".txt")
                
                base_script.run_network(prev, network,
                                          nts[i], pref_file,
                                          output_dir, create_animation=new_network, step=1000, create_picture=new_network, tag=f"_{networks_strings[i]}")
                prev = network

            print(f"✓ Completed gamma = {gamma_val:.4f}, gamma2 = {gamma2_val:.4f}")
        except Exception as e:
            print(f"✗ Error with gamma = {gamma_val:.4f}, gamma2 = {gamma2_val:.4f}")
        print(f"Results saved to: {output_dir}")
    
# Define values to test
gamma_values = np.repeat(.01, 16)
gamma_values = np.append(gamma_values, [.0375]*14)
gamma_values = np.append(gamma_values, 1)

gamma2_values = np.linspace(.01, .15, 15)
gamma2_values = np.append(gamma2_values, 1)

gamma2_values = np.append(gamma2_values, .0375)
gamma2_values = np.append(gamma2_values, np.linspace(.04, .15, 12))
gamma2_values = np.append(gamma2_values, 1)

gamma2_values = np.append(gamma2_values, 1)

# first third
# gamma_values = gamma_values[:11]
# gamma2_values = gamma2_values[:11]
# second third
gamma_values = gamma_values[11:21]
gamma2_values = gamma2_values[11:21]
# last third
#gamma_values = gamma_values[21:]
#gamma2_values = gamma2_values[21:]

# ON 6 / 10 experiment
if __name__ == "__main__":
    print(f"Gamma values to test: {gamma_values}")
    print(f"Gamma2 values to test: {gamma2_values}")
    run_phase3_experiment(gamma_values, gamma2_values)
    print("\nSweep completed!")