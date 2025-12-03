import csv
import matplotlib.pyplot as plt
import sys
import os
import glob
import re
import numpy as np

## Global Variables
BASE_DIR = "phase3_experiments"  # which parent folder to look for results
DT = 0.1  # Time step for scaling values

# Global font size parameters - Modify these to change text sizes globally
GLOBAL_FONT_SIZE = 20
GLOBAL_TITLE_SIZE = 18
GLOBAL_LABEL_SIZE = 18
GLOBAL_TICK_SIZE = 18
GLOBAL_LEGEND_SIZE = 16

def read_csv_by_timestep(csv_path, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Read CSV data and return all data points.
    This is used to collect all data first, then equidistant sampling will be done
    at the cumulative level across all CSVs.
    
    Args:
        csv_path (str): Path to the CSV file
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Not used in this function, kept for compatibility
        
    Returns:
        tuple: (time_values, data_values, column_names) or (None, None, None) if error
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Read header
            columns = next(reader)
            # Strip whitespace from column names
            columns = [col.strip() for col in columns]
            
            if len(columns) <= max(time_col_index, value_col_index):
                print(f"Error: CSV file must have at least {max(time_col_index, value_col_index)+1} columns. Found {len(columns)} columns in {csv_path}")
                return None, None, None
            
            # Read all data rows
            time_values = []
            data_values = []
            for row in reader:
                if len(row) > max(time_col_index, value_col_index):  # Ensure row has enough columns
                    try:
                        time_value = float(row[time_col_index].strip())
                        data_value = float(row[value_col_index].strip())
                        time_values.append(time_value)
                        data_values.append(data_value)
                    except ValueError:
                        continue  # Skip invalid rows
            
            if not time_values:
                print(f"Error: No valid data rows found in {csv_path}")
                return None, None, None
            
            return time_values, data_values, columns
        
    except FileNotFoundError:
        print(f"Warning: File '{csv_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None, None, None

def create_equidistant_cumulative_data(all_cumulative_times, all_cumulative_values, timestep_interval=1000):
    """
    Create equidistant points from cumulative data at exact multiples of timestep_interval.
    
    Args:
        all_cumulative_times (list): All cumulative time points
        all_cumulative_values (list): All cumulative value points
        timestep_interval (float): Interval for equidistant sampling
        
    Returns:
        tuple: (equidistant_times, equidistant_values)
    """
    if not all_cumulative_times or not all_cumulative_values:
        return [], []
    
    min_time = min(all_cumulative_times)
    max_time = max(all_cumulative_times)
    
    # Create data interpolator
    import numpy as np
    
    # Sort by time to ensure proper interpolation
    sorted_data = sorted(zip(all_cumulative_times, all_cumulative_values))
    sorted_times = [t for t, v in sorted_data]
    sorted_values = [v for t, v in sorted_data]
    
    # Create equidistant time points starting from the first multiple of timestep_interval
    # that is >= min_time
    first_interval = int(np.ceil(min_time / timestep_interval)) * timestep_interval
    
    equidistant_times = []
    current_time = first_interval
    while current_time <= max_time:
        equidistant_times.append(current_time)
        current_time += timestep_interval
    
    # Interpolate values at equidistant time points
    if equidistant_times:
        equidistant_values = np.interp(equidistant_times, sorted_times, sorted_values)
        return equidistant_times, equidistant_values.tolist()
    else:
        return [], []

def read_csv_last_row_values(csv_path, time_col_index=0, value_col_index=1):
    """
    Read CSV data and return the last row values for specified columns.
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Read header
            columns = next(reader)
            # Strip whitespace from column names
            columns = [col.strip() for col in columns]
            
            if len(columns) <= max(time_col_index, value_col_index):
                print(f"Error: CSV file must have at least {max(time_col_index, value_col_index)+1} columns. Found {len(columns)} columns in {csv_path}")
                return None, None, None
            
            # Read all data rows
            last_row = None
            for row in reader:
                if len(row) > max(time_col_index, value_col_index):  # Ensure row has enough columns
                    last_row = row
            
            if last_row is None:
                print(f"Error: No valid data rows found in {csv_path}")
                return None, None, None
            
            try:
                # Get the values from the last row
                time_value = float(last_row[time_col_index].strip())
                data_value = float(last_row[value_col_index].strip())
                return time_value, data_value, columns
            except ValueError:
                print(f"Error: Could not convert last row values to float in {csv_path}")
                return None, None, None
        
    except FileNotFoundError:
        print(f"Warning: File '{csv_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None, None, None

def plot_gamma2_comparison(gamma1_value, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for different gamma2 values with points at regular timestep intervals.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0100")
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma1_value}_gamma2_*"
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract gamma2 values and collect data
    gamma2_data = {}
    
    for directory in matching_dirs:
        # Extract gamma2 value from directory name
        match = re.search(r'gamma2_([0-9.]+)', os.path.basename(directory))
        if not match:
            continue
            
        gamma2_val = match.group(1)
        
        # Network types in order for cumulative plotting - Only PM networks 2-5
        network_types = ['PM 2', 'PM 3', 'PM 4', 'PM 5']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV (PM 2) - use all data points from beginning
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # Add all points from the first CSV starting from time 0
                        for time_val, data_val in zip(time_vals, data_vals):
                            all_cumulative_times.append(time_val)
                            all_cumulative_values.append(data_val)
                        
                        # Update cumulative counters with final values
                        cumulative_time = time_vals[-1]
                        cumulative_value = data_vals[-1]
                        print(f"Gamma2={gamma2_val}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in main function)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
                            print(f"Gamma2={gamma2_val}, {network_type}: time={final_time}, value={final_data}, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                gamma2_data[gamma2_val] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"Gamma2={gamma2_val}: Created {len(equidistant_times)} equidistant points")
                print(f"Gamma2={gamma2_val}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not gamma2_data:
        print("No valid data found to plot.")
        return
    
    # Sort gamma2 values for consistent ordering
    sorted_gamma2_values = sorted(gamma2_data.keys(), key=float)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Set larger font sizes for better readability in papers
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(sorted_gamma2_values)))
    
    # Collect transition points for vertical lines (using first gamma2 value as reference)
    if sorted_gamma2_values:
        reference_data = gamma2_data[sorted_gamma2_values[0]]
        network_types = ['PM 2', 'PM 3', 'PM 4', 'PM 5']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference gamma2, recalculate transition points
        ref_gamma2_val = sorted_gamma2_values[0]
        directory = None
        for dir_path in matching_dirs:
            if f"gamma2_{ref_gamma2_val}" in dir_path:
                directory = dir_path
                break
        
        if directory:
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV (PM 2)
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take only the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Plot each gamma2 value as a separate line
    for i, gamma2_val in enumerate(sorted_gamma2_values):
        data = gamma2_data[gamma2_val]
        plt.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$\\gamma_2$ = {gamma2_val}', alpha=0.8)
    
    # Add vertical lines and labels for network transitions
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = plt.ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            plt.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=16, alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in gamma2_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        # Clean up column names for professional display
        def format_label(label):
            if 'time_seconds' in label.lower():
                return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
            return label.replace('_', ' ').title()
        
        x_label = f'Cumulative {format_label(column_names[time_col_index])}'
        y_label = f'Cumulative {format_label(column_names[value_col_index])}'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(f'{y_label} vs {x_label} for Different $\\gamma_2$ Values ($\\gamma_1$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        plt.suptitle(f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index} for Different $\\gamma_2$ Values ($\\gamma$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(top=0.9)  # Adjust top margin for title
    plt.tight_layout(rect=(0, 0, 1, 0.92))  # Leave more space for title at top

    # Save the plot
    if column_names:
        # Build directory first
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_plots", f"gamma2_comparison_detailed_cumulative_{column_names[time_col_index]}_vs_cumulative_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for $\\gamma$ = {gamma1_value}")
    print(f"Found {len(sorted_gamma2_values)} gamma2 values: {sorted_gamma2_values}")

def plot_specific_gamma2_comparison(gamma1_value, gamma2_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for specific gamma2 values with points at regular timestep intervals.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0100")
        gamma2_values (list): List of gamma2 values to plot (e.g., ["0.0100", "0.0500", "0.1000"])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    
    # Collect data for specified gamma2 values
    gamma2_data = {}
    
    # Sort gamma2 values for consistent ordering
    sorted_gamma2_values = sorted([float(g) for g in gamma2_values])
    
    for gamma2_val in sorted_gamma2_values:
        gamma2_str = f"{gamma2_val:.4f}"
        directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if not os.path.exists(directory):
            print(f"Warning: No directory found for gamma1 = {gamma1_value}, gamma2 = {gamma2_str}")
            continue
        
        # Network types in order for cumulative plotting - Only PM networks 2-5
        network_types = ['PM 2', 'PM 3', 'PM 4', 'PM 5']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV (PM 2) - use all data points from beginning
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # Add all points from the first CSV starting from time 0
                        for time_val, data_val in zip(time_vals, data_vals):
                            all_cumulative_times.append(time_val)
                            all_cumulative_values.append(data_val)
                        
                        # Update cumulative counters with final values
                        cumulative_time = time_vals[-1]
                        cumulative_value = data_vals[-1]
                        print(f"Gamma2={gamma2_str}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in main function)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
                            print(f"Gamma2={gamma2_str}, {network_type}: time={final_time}, value={final_data}, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                gamma2_data[gamma2_str] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"Gamma2={gamma2_str}: Created {len(equidistant_times)} equidistant points")
                print(f"Gamma2={gamma2_str}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not gamma2_data:
        print("No valid data found to plot.")
        return
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Set larger font sizes for better readability in papers
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(gamma2_data)))
    
    # Collect transition points for vertical lines (using first gamma2 value as reference)
    if gamma2_data:
        sorted_items = sorted(gamma2_data.items(), key=lambda x: float(x[0]))
        reference_gamma2_str = sorted_items[0][0]
        network_types = ['PM 2', 'PM 3', 'PM 4', 'PM 5']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference gamma2, recalculate transition points
        directory_name = f"gamma_{gamma1_value}_gamma2_{reference_gamma2_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if os.path.exists(directory):
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV (PM 2)
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take only the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Plot each gamma2 value as a separate line
    for i, (gamma2_str, data) in enumerate(sorted(gamma2_data.items(), key=lambda x: float(x[0]))):
        plt.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$\\gamma_2$ = {gamma2_str}', alpha=0.8)
    
    # Add vertical lines and labels for network transitions
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = plt.ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            plt.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in gamma2_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        # Clean up column names for professional display
        def format_label(label):
            if 'time_seconds' in label.lower():
                return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
            return label.replace('_', ' ').title()
        
        x_label = f'Cumulative {format_label(column_names[time_col_index])}'
        y_label = f'Cumulative {format_label(column_names[value_col_index])}'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(f'{y_label} vs {x_label} - Selected $\\gamma_2$ Values ($\\gamma$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        plt.suptitle(f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index} - Selected $\\gamma_2$ Values ($\\gamma$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(top=0.9)  # Adjust top margin for title
    plt.tight_layout(rect=(0, 0, 1, 0.9))  # Leave more space for title at top

    # Save the plot
    if column_names:
        # Build directory first
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_plots", f"specific_gamma2_comparison_detailed_cumulative_{column_names[time_col_index]}_vs_cumulative_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {len(gamma2_data)} selected $\\gamma_2$ values")
    if gamma2_data:
        print(f"Plotted $\\gamma_2$ values: {list(gamma2_data.keys())}")

def plot_single_network_gamma2_comparison(gamma1_value, network_type, gamma2_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across different gamma2 values.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0100")
        network_type (str): Network type (e.g., "PM 5", "AM Base", etc.)
        gamma2_values (list): List of gamma2 values to plot (e.g., ["0.0100", "0.0500", "0.1000"])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    
    # Collect data for specified gamma2 values
    gamma2_data = {}
    
    # Sort gamma2 values for consistent ordering
    sorted_gamma2_values = sorted([float(g) for g in gamma2_values])
    
    for gamma2_val in sorted_gamma2_values:
        gamma2_str = f"{gamma2_val:.4f}"
        directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if not os.path.exists(directory):
            print(f"Warning: No directory found for gamma1 = {gamma1_value}, gamma2 = {gamma2_str}")
            continue
        
        csv_filename = f"time_series_data_{network_type}.csv"
        csv_path = os.path.join(directory, csv_filename)
        
        if os.path.exists(csv_path):
            # Read all data points for this network
            time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
            if time_vals is not None and data_vals is not None:
                # Create equidistant sampling if needed
                if timestep_interval > 100:  # Only do equidistant sampling for larger intervals
                    equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                        time_vals, data_vals, timestep_interval)
                    if equidistant_times:
                        gamma2_data[gamma2_str] = {
                            'times': equidistant_times,
                            'values': equidistant_values,
                            'column_names': cols
                        }
                        print(f"Gamma2={gamma2_str}, {network_type}: Created {len(equidistant_times)} equidistant points")
                        print(f"Gamma2={gamma2_str}: First 10 times: {equidistant_times[:10]}")
                else:
                    # Use all data points for smaller intervals
                    gamma2_data[gamma2_str] = {
                        'times': time_vals,
                        'values': data_vals,
                        'column_names': cols
                    }
                    print(f"Gamma2={gamma2_str}, {network_type}: Using all {len(time_vals)} data points")
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    if not gamma2_data:
        print("No valid data found to plot.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set larger font sizes for better readability in papers
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(gamma2_data)))
    
    # Plot each gamma2 value as a separate line
    for i, (gamma2_str, data) in enumerate(sorted(gamma2_data.items(), key=lambda x: float(x[0]))):
        plt.plot(data['times'], data['values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$\\gamma_2$ = {gamma2_str}', alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in gamma2_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        # Clean up column names for professional display
        def format_label(label):
            if 'time_seconds' in label.lower():
                return label.replace('time_seconds', 'Time (seconds)').replace('Time_Seconds', 'Time (seconds)')
            return label.replace('_', ' ').title()
        
        x_label = format_label(column_names[time_col_index])
        y_label = format_label(column_names[value_col_index])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(f'{y_label} vs {x_label} for {network_type} - Different $\\gamma_2$ Values ($\\gamma$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    else:
        plt.xlabel(f'Column {time_col_index}')
        plt.ylabel(f'Column {value_col_index}')
        plt.suptitle(f'Column {value_col_index} vs Column {time_col_index} for {network_type} - Different $\\gamma_2$ Values ($\\gamma$ = {gamma1_value})', x=0.5, y=0.95, fontsize=GLOBAL_TITLE_SIZE)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(top=0.9)  # Adjust top margin for title
    plt.tight_layout(rect=(0, 0, 1, 0.9))  # Leave more space for title at top

    # Save the plot
    if column_names:
        # Build directory first
        safe_network_name = network_type.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_plots", f"single_network_{safe_network_name}_gamma2_comparison_{column_names[time_col_index]}_vs_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {network_type} with {len(gamma2_data)} $\\gamma_2$ values")
    if gamma2_data:
        print(f"Plotted $\\gamma_2$ values: {list(gamma2_data.keys())}")

def plot_all_gamma2_single_network(gamma1_value, network_type, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across ALL available gamma2 values.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0100")
        network_type (str): Network type (e.g., "PM 5", "AM Base", etc.)
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma1_value}_gamma2_*"
    
    # Find all matching directories for this gamma1 value
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract gamma2 values from directory names
    gamma2_values = []
    for directory in matching_dirs:
        match = re.search(r'gamma2_([0-9.]+)', os.path.basename(directory))
        if match:
            # Check if the specific network CSV exists in this directory
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(directory, csv_filename)
            if os.path.exists(csv_path):
                gamma2_values.append(match.group(1))
    
    if not gamma2_values:
        print(f"No valid gamma2 directories with {network_type} data found for gamma1 = {gamma1_value}")
        return
    
    print(f"Found {len(gamma2_values)} gamma2 values for {network_type}: {sorted(gamma2_values, key=float)}")
    
    # Use the existing single network plotting function
    plot_single_network_gamma2_comparison(gamma1_value, network_type, gamma2_values, time_col_index, value_col_index, timestep_interval)

def main():
    """
    Main function to handle command line arguments or interactive input.
    """
    if len(sys.argv) >= 2:
        if sys.argv[1] == "specific":
            # Specific gamma2 values mode: python script.py specific gamma1_value gamma2_1,gamma2_2,gamma2_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 4:
                print("Error: Please provide gamma1 value and gamma2 values. Usage: python script.py specific 0.0100 0.0100,0.0500,0.1000 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            gamma2_str = sys.argv[3]
            gamma2_values = [g.strip() for g in gamma2_str.split(',')]
            time_col_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            value_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            timestep_interval = int(float(sys.argv[6])) if len(sys.argv) > 6 else 1000
            
            plot_specific_gamma2_comparison(gamma1_value, gamma2_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single":
            # Single network mode: python script.py single gamma1_value "network_type" gamma2_1,gamma2_2,gamma2_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 5:
                print("Error: Please provide gamma1 value, network type, and gamma2 values. Usage: python script.py single 0.0100 \"PM 5\" 0.0100,0.0500,0.1000 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            network_type = sys.argv[3]
            gamma2_str = sys.argv[4]
            gamma2_values = [g.strip() for g in gamma2_str.split(',')]
            time_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            value_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 1
            timestep_interval = int(float(sys.argv[7])) if len(sys.argv) > 7 else 1000
            
            plot_single_network_gamma2_comparison(gamma1_value, network_type, gamma2_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single_all":
            # Single network ALL gamma2 mode: python script.py single_all gamma1_value "network_type" [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 4:
                print("Error: Please provide gamma1 value and network type. Usage: python script.py single_all 0.0100 \"PM 5\" [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            network_type = sys.argv[3]
            time_col_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            value_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            timestep_interval = int(float(sys.argv[6])) if len(sys.argv) > 6 else 1000
            
            plot_all_gamma2_single_network(gamma1_value, network_type, time_col_index, value_col_index, timestep_interval)
        else:
            # Gamma2 comparison mode
            gamma1_value = sys.argv[1]
            time_col_index = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
            value_col_index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
            timestep_interval = int(float(sys.argv[4])) if len(sys.argv) > 4 else 1000
            plot_gamma2_comparison(gamma1_value, time_col_index, value_col_index, timestep_interval)
    else:
        # Interactive mode
        print("Choose mode:")
        print("1. Plot comparison for all gamma2 values for a gamma1")
        print("2. Plot comparison for specific gamma2 values for a gamma1")
        print("3. Plot comparison for a single network type across specific gamma2 values")
        print("4. Plot comparison for a single network type across ALL gamma2 values")
        
        choice = input("Enter choice (1, 2, 3, or 4): ").strip()
        
        if choice == "1":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0100): ").strip()
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_gamma2_comparison(gamma1_value, time_col_index, value_col_index, timestep_interval)
        elif choice == "2":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0100): ").strip()
            gamma2_str = input("Enter gamma2 values separated by commas (e.g., 0.0100, 0.0500, 0.1000): ").strip()
            gamma2_values = [g.strip() for g in gamma2_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_specific_gamma2_comparison(gamma1_value, gamma2_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "3":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0100): ").strip()
            network_type = input("Enter network type (e.g., 'PM 5', 'AM Base', 'AM 2', etc.): ").strip()
            gamma2_str = input("Enter gamma2 values separated by commas (e.g., 0.0100, 0.0500, 0.1000): ").strip()
            gamma2_values = [g.strip() for g in gamma2_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_single_network_gamma2_comparison(gamma1_value, network_type, gamma2_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "4":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0100): ").strip()
            network_type = input("Enter network type (e.g., 'PM 5', 'AM Base', 'AM 2', etc.): ").strip()
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_all_gamma2_single_network(gamma1_value, network_type, time_col_index, value_col_index, timestep_interval)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()