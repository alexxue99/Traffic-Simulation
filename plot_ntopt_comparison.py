import csv
import matplotlib.pyplot as plt
import sys
import os
import glob
import re

## Global Variables
BASE_DIR = "phase1_experiments" # which parent folder to look for results
DT = 0.1  # Time step for scaling nt_opt values


def read_csv_columns(csv_path, col1_index=0, col2_index=1):
    """
    Read CSV data and return specified columns and column names.
    If the first column restarts (goes back to 0), only return data from the last restart.
    
    Args:
        csv_path (str): Path to the CSV file
        col1_index (int): Index of first column (default: 0)
        col2_index (int): Index of second column (default: 1)
        
    Returns:
        tuple: (x_data, y_data, columns) or (None, None, None) if error
    """
    try:
        x_data_all = []
        y_data_all = []
        columns = []
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Read header
            columns = next(reader)
            # Strip whitespace from column names
            columns = [col.strip() for col in columns]
            
            if len(columns) <= max(col1_index, col2_index):
                print(f"Error: CSV file must have at least {max(col1_index, col2_index)+1} columns. Found {len(columns)} columns in {csv_path}")
                return None, None, None
            
            # Read data rows
            for row in reader:
                if len(row) > max(col1_index, col2_index):  # Ensure row has enough columns
                    try:
                        x_val = float(row[col1_index].strip())
                        y_val = float(row[col2_index].strip())
                        x_data_all.append(x_val)
                        y_data_all.append(y_val)
                    except ValueError:
                        continue  # Skip invalid rows silently
        
        if not x_data_all or not y_data_all:
            print(f"Error: No valid numeric data found in {csv_path}")
            return None, None, None
        
        # Find the last restart point (where x_data goes back to 0 or near 0)
        last_restart_idx = 0
        for i in range(1, len(x_data_all)):
            # If current value is significantly smaller than previous (indicating restart)
            if x_data_all[i] < x_data_all[i-1] and x_data_all[i] <= 1.0:  # Allow small tolerance for restart detection
                last_restart_idx = i
        
        # Return data from the last restart point
        x_data = x_data_all[last_restart_idx:]
        y_data = y_data_all[last_restart_idx:]
        
        return x_data, y_data, columns
        
    except FileNotFoundError:
        print(f"Warning: File '{csv_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None, None, None

def plot_gamma_ntopt_comparison(gamma_value, col1_index=0, col2_index=1):
    """
    Plot time series for all nt_opt values for a given gamma.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        col1_index (int): Index of first column (x-axis)
        col2_index (int): Index of second column (y-axis)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma_value}_ntopt_*"
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(range(len(matching_dirs)))  # Use different colors
    
    plotted_any = False
    columns = None
    
    # Sort directories by nt_opt value for consistent ordering
    def extract_ntopt(dirname):
        match = re.search(r'ntopt_(\d+)', dirname)
        return int(match.group(1)) if match else 0
    
    matching_dirs.sort(key=extract_ntopt)
    
    for i, directory in enumerate(matching_dirs):
        # Extract nt_opt value from directory name
        match = re.search(r'ntopt_(\d+)', os.path.basename(directory))
        if not match:
            continue
            
        nt_opt = match.group(1)
        csv_path = os.path.join(directory, "time_series_data.csv")
        
        x_data, y_data, file_columns = read_csv_columns(csv_path, col1_index, col2_index)
        
        if x_data is not None and y_data is not None:
            scaled_nt_opt = int(int(nt_opt) * DT)
            plt.plot(x_data, y_data, color=colors[i], linewidth=1.5,
                    label=f'$nt_{{opt}}$ = {scaled_nt_opt}', marker='o', markersize=2)
            plotted_any = True
            
            if columns is None:
                columns = file_columns
            
            print(f"Plotted: {directory} (ntopt = {nt_opt}, {len(x_data)} points)")
    
    if not plotted_any:
        print("No valid data found to plot.")
        return
    
    # Set labels and title
    if columns:
        # Clean up column names for professional display
        def format_label(label):
            if 'time_seconds' in label.lower():
                return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
            return label.replace('_', ' ').title()
        
        x_label = format_label(columns[col1_index])
        y_label = format_label(columns[col2_index])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{y_label} vs {x_label} ($\\gamma$ = {gamma_value})')
    else:
        plt.xlabel(f'Column {col1_index}')
        plt.ylabel(f'Column {col2_index}')
        plt.title(f'Column {col2_index} vs Column {col1_index} ($\\gamma$ = {gamma_value})')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # save the plot
    if columns:
        # Build directory first (just gamma folder)
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"all_ntopts_{columns[col1_index]}_vs_{columns[col2_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for $\\gamma$ = {gamma_value}")
    print(f"Found {len([d for d in matching_dirs if os.path.exists(os.path.join(d, 'time_series_data.csv'))])} valid datasets")

def plot_specific_ntopt_comparison(gamma_value, ntopt_values, col1_index=0, col2_index=1):
    """
    Plot time series for specific nt_opt values for a given gamma.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        ntopt_values (list): List of nt_opt values to plot (e.g., [10, 100, 600])
        col1_index (int): Index of first column (x-axis)
        col2_index (int): Index of second column (y-axis)
    """
    base_dir = BASE_DIR
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(range(len(ntopt_values)))  # Use different colors
    
    plotted_any = False
    columns = None
    valid_ntopts = []
    
    # Sort ntopt values for consistent ordering
    ntopt_values_sorted = sorted([int(n) for n in ntopt_values])
    
    for i, ntopt_val in enumerate(ntopt_values_sorted):
        # Look for directory with this gamma and ntopt combination
        directory_name = f"gamma_{gamma_value}_ntopt_{ntopt_val}"
        directory = os.path.join(base_dir, directory_name)
        
        if not os.path.exists(directory):
            print(f"Warning: No directory found for gamma = {gamma_value}, ntopt = {ntopt_val} in base directory {base_dir}")
            continue
            
        csv_path = os.path.join(directory, "time_series_data.csv")
        
        x_data, y_data, file_columns = read_csv_columns(csv_path, col1_index, col2_index)
        
        if x_data is not None and y_data is not None:
            scaled_nt_opt = int(ntopt_val * DT)
            plt.plot(x_data, y_data, color=colors[i], linewidth=1.5,
                    label=f'$nt_{{opt}}$ = {scaled_nt_opt}', alpha=0.8, marker='o', markersize=3)
            plotted_any = True
            valid_ntopts.append(str(ntopt_val))
            
            if columns is None:
                columns = file_columns
            
            print(f"Plotted: {directory} (ntopt = {ntopt_val}, {len(x_data)} points)")
    
    if not plotted_any:
        print("No valid data found to plot.")
        return
    
    # Set labels and title
    if columns:
        # Clean up column names for professional display
        def format_label(label):
            if 'time_seconds' in label.lower():
                return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
            return label.replace('_', ' ').title()
        
        x_label = format_label(columns[col1_index])
        y_label = format_label(columns[col2_index])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{y_label} vs {x_label} - Selected $nt_{{opt}}$ Values ($\\gamma$ = {gamma_value})')
    else:
        plt.xlabel(f'Column {col1_index}')
        plt.ylabel(f'Column {col2_index}')
        plt.title(f'Column {col2_index} vs Column {col1_index} - Selected $nt_{{opt}}$ Values ($\\gamma$ = {gamma_value})')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # save the plot
    if columns:
        # Build directory first (just gamma folder)
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"specific_ntopts_{columns[col1_index]}_vs_{columns[col2_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()

    print(f"\nSuccessfully plotted comparison for {len(valid_ntopts)} selected $n_{{\\text{{opt}}}}$ values")
    if valid_ntopts:
        print(f"Plotted $n_{{\\text{{opt}}}}$ values: {', '.join(valid_ntopts)}")

def plot_single_csv_columns(csv_path, col1_index=0, col2_index=1):
    """
    Plot specified columns from a single CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' does not exist.")
        return
        
    x_data, y_data, columns = read_csv_columns(csv_path, col1_index, col2_index)
    
    if x_data is None:
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'b-', linewidth=1.5)
    
    # Set labels and title
    def format_label(label):
        if 'time_seconds' in label.lower():
            return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
        return label.replace('_', ' ').title()
    
    x_label = format_label(columns[col1_index])
    y_label = format_label(columns[col2_index])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    print(f"Successfully plotted data from: {csv_path}")
    print(f"X-axis: {columns[col1_index]} ({len(x_data)} data points)")
    print(f"Y-axis: {columns[col2_index]}")

def main():
    """
    Main function to handle command line arguments or interactive input.
    """
    if len(sys.argv) >= 2:
        if sys.argv[1].endswith('.csv') or '/' in sys.argv[1] or '\\' in sys.argv[1]:
            # Single CSV file mode
            col1_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            col2_index = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            plot_single_csv_columns(sys.argv[1], col1_index, col2_index)
        elif sys.argv[1] == "specific":
            # Specific ntopt values mode: python script.py specific gamma_value ntopt1,ntopt2,ntopt3 [col1] [col2]
            if len(sys.argv) < 4:
                print("Error: Please provide gamma value and ntopt values. Usage: python script.py specific 0.0375 10,100,600 [col1] [col2]")
                return
            
            gamma_value = sys.argv[2]
            ntopt_str = sys.argv[3]
            ntopt_values = [int(n.strip()) for n in ntopt_str.split(',')]
            col1_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            col2_index = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            
            plot_specific_ntopt_comparison(gamma_value, ntopt_values, col1_index, col2_index)
        else:
            # Gamma ntopt comparison mode
            gamma_value = sys.argv[1]
            col1_index = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
            col2_index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
            plot_gamma_ntopt_comparison(gamma_value, col1_index, col2_index)
    else:
        # Interactive mode
        print("Choose mode:")
        print("1. Plot single CSV file")
        print("2. Plot comparison for all ntopt values for a gamma")
        print("3. Plot comparison for specific ntopt values for a gamma")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            csv_path = input("Enter the path to the CSV file: ").strip()
            col1_str = input("Enter first column index (default 0): ").strip()
            col2_str = input("Enter second column index (default 1): ").strip()
            
            col1_index = int(col1_str) if col1_str else 0
            col2_index = int(col2_str) if col2_str else 1
            
            plot_single_csv_columns(csv_path, col1_index, col2_index)
        elif choice == "2":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            col1_str = input("Enter first column index (default 0): ").strip()
            col2_str = input("Enter second column index (default 1): ").strip()
            
            col1_index = int(col1_str) if col1_str else 0
            col2_index = int(col2_str) if col2_str else 1
            
            plot_gamma_ntopt_comparison(gamma_value, col1_index, col2_index)
        elif choice == "3":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            ntopt_str = input("Enter ntopt values separated by commas (e.g., 10, 100, 600, 6000): ").strip()
            ntopt_values = [int(n.strip()) for n in ntopt_str.split(',')]
            
            col1_str = input("Enter first column index (default 0): ").strip()
            col2_str = input("Enter second column index (default 1): ").strip()
            
            col1_index = int(col1_str) if col1_str else 0
            col2_index = int(col2_str) if col2_str else 1
            
            plot_specific_ntopt_comparison(gamma_value, ntopt_values, col1_index, col2_index)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()