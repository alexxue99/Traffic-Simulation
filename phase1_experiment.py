import os

from am_networks_setup import setup_am_networks
import optimization_script_parallel
import base_script

def run_phase1_experiment(gamma_values, nt_opt_values):
    """
    Optimizes the base AM network using the inputted gamma values and nt opt values.

    Runs the network using the optimized parameters, and creates animation and pictures.
    
    Input 0 for an nt_opt_value to skip optimization and just run the network.
    
    Stores results to phase1_experiments/.
    """
    
    print("Starting Parameters Sweep")
    print("=" * 50)
    
    for i, gamma_val in enumerate(gamma_values):
        nt_opt_val = nt_opt_values[i]
        print(f"\nExperiment {i+1}/{len(gamma_values)}: gamma = {gamma_val:.4f}")
        print(f"\nExperiment {i+1}/{len(gamma_values)}: nt_opt_val = {nt_opt_val:d}")
        print("-" * 30)
        
        # Steps 1-2: instead of reloading, just reset everything by constructing using the new gamma
        networks = setup_am_networks(gamma_val)
        
        # Step 3: Create output directory for this gamma
        gamma_str = f"{gamma_val:.4f}"
        output_dir = os.path.join("phase1_experiments", f"gamma_{gamma_str}_ntopt_{nt_opt_val}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        network_am_base = networks[0]  # Get the updated network
                
        # Step 4: Run the optimization with output directory
        print("Starting optimization...")
        if nt_opt_val == 0:
            run_optimization = False
        else:
            run_optimization = True

        try:
            if run_optimization:
                # Run with output directory using minimal parallel version
                optimization_script_parallel.run_optimization(None, network_am_base,
                                        nt_opt_val, output_dir, by_data=False)
                print(f"✓ Completed gamma = {gamma_val:.4f}, nt_opt = {nt_opt_val}")
        except Exception as e:
            print(f"✗ Error with gamma = {gamma_val:.4f}, nt_opt = {nt_opt_val}: {e}")
        print(f"Results saved to: {output_dir}")

        # Step 5: Run the network with optimal parameters
        print("Running network and storing data...")
        networks = setup_am_networks(gamma_val)  # Reset networks to ensure clean state
        network_am_base = networks[0]
        if run_optimization:
            base_script.run_network(None, network_am_base,
                                        87000, os.path.join(output_dir, "opt_parameters.txt"),
                                        output_dir, create_animation=True, step=300, create_picture=True)
        else:
            base_script.run_network(None, network_am_base,
                                        87000, None,
                                        output_dir, create_animation=True, step=300, create_picture=True)
        
        
# Define values to test
gamma_values = [i for i in [0.01, 0.0375, 0.075, 0.125, 1.0] for _ in range(5)]  # Repeat each 5 times
nt_opt_values = [0, 10, 100, 600, 6000] * 5  # Repeat the sequence 5 times

if __name__ == "__main__":
    print(f"Gamma values to test: {gamma_values}")
    print(f"nt_opt values to test: {nt_opt_values}")
    run_phase1_experiment(gamma_values, nt_opt_values)
    print("\nSweep completed!")