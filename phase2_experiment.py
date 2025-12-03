import os

from am_networks_setup import setup_am_networks
from pm_networks_setup import setup_pm_networks
import optimization_script_parallel
import base_script

import pandas as pd
import numpy as np
import glob

def run_phase2_experiment(gamma_values, nt_opt_values, num_trials):
    """
    Optimizes the networks from the base AM network to PM base network, using the inputted gamma values and nt opt values.

    Runs the networks using the optimized parameters, and creates animation and pictures. Averages results over num_trials runs.
    
    Input 0 for an nt_opt_value to skip optimization and just run the network.
    
    Stores results to phase2_experiments/.
    """
    
    print("Starting Parameters Sweep")
    print("=" * 50)
    
    for i, gamma_val in enumerate(gamma_values):
        nt_opt_val = nt_opt_values[i]
        print(f"\nExperiment {i+1}/{len(gamma_values)}: gamma = {gamma_val:.4f}")
        print(f"\nExperiment {i+1}/{len(gamma_values)}: nt_opt_val = {nt_opt_val:d}")
        print("-" * 30)

        # Step 3: Create output directory for this gamma
        gamma_str = f"{gamma_val:.4f}"
        output_dir = os.path.join("phase2_experiments_trials", f"gamma_{gamma_str}_ntopt_{nt_opt_val}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Step 4: Run trials with fresh networks
        print("Starting trials...")

        try:
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials} for gamma = {gamma_val:.4f}, nt_opt = {nt_opt_val}")

                # Create fresh networks for each trial to avoid state contamination
                network_am_base, network_am_2, network_am_3 = setup_am_networks(gamma_val)
                network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5 = setup_pm_networks(gamma_val)

                networks = (network_am_base, network_am_2, network_am_3,
                           network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5)
                networks_strings = ("AM Base", "AM 2", "AM 3",
                                   "PM Base", "PM 2", "PM 3", "PM 4", "PM 5")
                nts = (87000, 33600, 20400, 18600, 37800, 4200, 6600, 13800)

                prev = None
                for i, network in enumerate(networks):
                    if i >= 4:
                        continue  # Skip PM 2 network and further networks for phase 2 experiments

                    network.reset_by_data()

                    if nt_opt_val != 0:
                        optimization_script_parallel.run_optimization(prev, network,
                                                nt_opt_val, output_dir, tag=f"_{networks_strings[i]}_Trial {trial + 1}")
                        network.reset_by_data()
                    
                    pref_file = None if nt_opt_val == 0 else os.path.join(output_dir, "opt_parameters" + f"_{networks_strings[i]}_Trial {trial + 1}" + ".txt")
                    base_script.run_network(prev, network,
                                            nts[i], pref_file,
                                            output_dir, create_animation=False, step=1000, create_picture=False, tag=f"_{networks_strings[i]}_Trial {trial + 1}")
                    prev = network

            # Recreate networks for averaging (networks were created inside trial loop)
            network_am_base, network_am_2, network_am_3 = setup_am_networks(gamma_val)
            network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5 = setup_pm_networks(gamma_val)
            networks = (network_am_base, network_am_2, network_am_3,
                       network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5)
            networks_strings = ("AM Base", "AM 2", "AM 3",
                               "PM Base", "PM 2", "PM 3", "PM 4", "PM 5")

            ## Averaging Step: Average over all parameters (under .txt) and over all time series data (under .csv)

            for i, network in enumerate(networks):
                if i >= 4:
                    continue

                # Read results from all trials and average them
                # Step 1: List all files (adjust the path or pattern as needed)
                path = os.path.join(output_dir, f"*_{networks_strings[i]}_Trial*.csv")
                csv_files = glob.glob(path)
                txt_files = glob.glob(path.replace(".csv", ".txt"))

                # Step 2: Read the files
                dfs = [pd.read_csv(f) for f in csv_files]
                arr = [np.loadtxt(f) for f in txt_files]

                # Check if all have same shape
                shapes = [df.shape for df in dfs]
                shapes2 = [a.shape for a in arr]
                if len(set(shapes)) != 1 or len(set(shapes2)) != 1:
                    print(shapes)
                    print(shapes2)
                    print(networks_strings[i])
                    raise ValueError("Not all CSV files have the same shape!")

                # Step 3: Convert to numpy arrays and average elementwise
                arrays = [df.to_numpy() for df in dfs]
                avg_array_csv = np.mean(arrays, axis=0)
                avg_array_txt = np.mean(arr, axis=0)

                # Step 4: Recreate a DataFrame with the same headers and index
                avg_df = pd.DataFrame(avg_array_csv, columns=dfs[0].columns)

                # Step 5: Save the averaged result
                csv_file_path = os.path.join(output_dir, "time_series_data" + f"_{networks_strings[i]}_Averaged" + ".csv")
                avg_df.to_csv(csv_file_path, index=False)
                txt_file_path = os.path.join(output_dir, "opt_parameters" + f"_{networks_strings[i]}_Averaged" + ".txt")
                np.savetxt(txt_file_path, avg_array_txt)
                
            print(f"✓ Completed gamma = {gamma_val:.4f}, nt_opt = {nt_opt_val}")
        except Exception as e:
            print(f"✗ Error with gamma = {gamma_val:.4f}, nt_opt = {nt_opt_val}: {e}")
        print(f"Results saved to: {output_dir}")
    
# Define values to test - 4 configurations in a single run
gamma_values = [0.01, 0.01, 0.0375, 0.0375]
nt_opt_values = [10, 600, 10, 600]

if __name__ == "__main__":
    print(f"Gamma values to test: {gamma_values}")
    run_phase2_experiment(gamma_values, nt_opt_values, num_trials=10)
    print("\nSweep completed!")