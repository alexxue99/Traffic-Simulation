import csv
import matplotlib.pyplot as plt
import sys
import os
import glob
import re
import numpy as np

## Global Variables
BASE_DIR = "phase2_experiments_trials"  # which parent folder to look for results

# Global font size parameters - Modify these to change text sizes globally
GLOBAL_FONT_SIZE = 18
GLOBAL_TITLE_SIZE = 20
GLOBAL_LABEL_SIZE = 18
GLOBAL_TICK_SIZE = 14
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

def format_column_name(column_name):
    """
    Format column name by removing underscores and capitalizing words.
    
    Args:
        column_name (str): Original column name
        
    Returns:
        str: Formatted column name
    """
    # Replace underscores with spaces and capitalize each word
    formatted = ' '.join(word.capitalize() for word in column_name.split('_'))
    return formatted


def plot_ntopt_comparison(gamma_value, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for different nt_opt values with points at regular timestep intervals.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma_value}_ntopt_*"
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract nt_opt values and collect data
    ntopt_data = {}
    
    for directory in matching_dirs:
        # Extract nt_opt value from directory name
        match = re.search(r'ntopt_([0-9]+)', os.path.basename(directory))
        if not match:
            continue
            
        ntopt_val = match.group(1)
        
        # Phase 2 network types in order for cumulative plotting
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}_Averaged.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV - use all data points from beginning
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
                        print(f"nt_opt={ntopt_val}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in phase3)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
                            print(f"nt_opt={ntopt_val}, {network_type}: time={final_time}, value={final_data}, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                ntopt_data[ntopt_val] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"nt_opt={ntopt_val}: Created {len(equidistant_times)} equidistant points")
                print(f"nt_opt={ntopt_val}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not ntopt_data:
        print("No valid data found to plot.")
        return
    
    # Sort nt_opt values for consistent ordering
    sorted_ntopt_values = sorted(ntopt_data.keys(), key=int)
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(sorted_ntopt_values)))
    
    # Collect transition points for vertical lines (using first nt_opt value as reference)
    if sorted_ntopt_values:
        reference_data = ntopt_data[sorted_ntopt_values[0]]
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference nt_opt, recalculate transition points
        ref_ntopt_val = sorted_ntopt_values[0]
        directory = None
        for dir_path in matching_dirs:
            if f"ntopt_{ref_ntopt_val}" in dir_path:
                directory = dir_path
                break
        
        if directory:
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}_Averaged.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Plot each nt_opt value as a separate line
    for i, ntopt_val in enumerate(sorted_ntopt_values):
        data = ntopt_data[ntopt_val]
        plt.loglog(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'nt_opt = {ntopt_val}', alpha=0.8)
    
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
    for data in ntopt_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        plt.xlabel(f'Cumulative {formatted_time_col}')
        plt.ylabel(f'Cumulative {formatted_value_col}')
        plt.title(f'Cumulative {formatted_value_col} vs Cumulative {formatted_time_col} for Different nt_opt Values (gamma = {gamma_value})')
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        plt.title(f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index} for Different nt_opt Values (gamma = {gamma_value})')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        safe_time_col = formatted_time_col.replace(' ', '_')
        safe_value_col = formatted_value_col.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"ntopt_comparison_detailed_cumulative_{safe_time_col}_vs_cumulative_{safe_value_col}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for gamma = {gamma_value}")
    print(f"Found {len(sorted_ntopt_values)} nt_opt values: {sorted_ntopt_values}")

def plot_ntopt_comparison_with_differences(gamma_value, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves with comprehensive difference analysis using LOG-LOG plots for small differences.
    Creates multiple subplots to highlight percentage differences between ntopt values.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma_value}_ntopt_*"
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract nt_opt values and collect data
    ntopt_data = {}
    
    for directory in matching_dirs:
        # Extract nt_opt value from directory name
        match = re.search(r'ntopt_([0-9]+)', os.path.basename(directory))
        if not match:
            continue
            
        ntopt_val = match.group(1)
        
        # Phase 2 network types in order for cumulative plotting
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}_Averaged.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV - use all data points from beginning
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
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in phase3)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                ntopt_data[ntopt_val] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
    
    if not ntopt_data:
        print("No valid data found to plot.")
        return
    
    # Sort nt_opt values for consistent ordering
    sorted_ntopt_values = sorted(ntopt_data.keys(), key=int)
    
    # Use smallest ntopt as baseline
    baseline_ntopt = sorted_ntopt_values[0]
    baseline_data = ntopt_data[baseline_ntopt]
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    # Create comprehensive difference analysis with 4 subplots (LOG-LOG VERSION)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(sorted_ntopt_values)))
    
    # Collect transition points for vertical lines (using first nt_opt value as reference)
    if sorted_ntopt_values:
        reference_data = ntopt_data[sorted_ntopt_values[0]]
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference nt_opt, recalculate transition points
        ref_ntopt_val = sorted_ntopt_values[0]
        directory = None
        for dir_path in matching_dirs:
            if f"ntopt_{ref_ntopt_val}" in dir_path:
                directory = dir_path
                break
        
        if directory:
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}_Averaged.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Subplot 1: Original log-log plot
    for i, ntopt_val in enumerate(sorted_ntopt_values):
        data = ntopt_data[ntopt_val]
        ax1.loglog(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 1
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax1.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax1.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax1.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 2: Percentage difference from baseline
    for i, ntopt_val in enumerate(sorted_ntopt_values[1:], 1):  # Skip baseline
        data = ntopt_data[ntopt_val]
        baseline_values_interp = np.interp(data['cumulative_times'],
                                         baseline_data['cumulative_times'],
                                         baseline_data['cumulative_values'])
        pct_diff = (np.array(data['cumulative_values']) - baseline_values_interp) / baseline_values_interp * 100
        
        ax2.plot(data['cumulative_times'], pct_diff,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$ vs $nt_{{opt}} = {baseline_ntopt}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 2
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax2.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax2.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax2.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 3: Normalized values (divide by baseline)
    for i, ntopt_val in enumerate(sorted_ntopt_values):
        data = ntopt_data[ntopt_val]
        if ntopt_val == baseline_ntopt:
            normalized_values = np.ones_like(data['cumulative_values'])  # Baseline = 1
        else:
            baseline_values_interp = np.interp(data['cumulative_times'],
                                             baseline_data['cumulative_times'],
                                             baseline_data['cumulative_values'])
            normalized_values = np.array(data['cumulative_values']) / baseline_values_interp
        
        ax3.plot(data['cumulative_times'], normalized_values,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 3
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax3.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax3.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax3.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 4: Absolute difference from baseline
    for i, ntopt_val in enumerate(sorted_ntopt_values[1:], 1):  # Skip baseline
        data = ntopt_data[ntopt_val]
        baseline_values_interp = np.interp(data['cumulative_times'],
                                         baseline_data['cumulative_times'],
                                         baseline_data['cumulative_values'])
        abs_diff = np.array(data['cumulative_values']) - baseline_values_interp
        
        ax4.plot(data['cumulative_times'], abs_diff,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val} - nt_{{opt}} = {baseline_ntopt}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 4
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax4.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax4.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax4.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Set titles and labels
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        ax1.set_title(f'Log-Log: Cumulative {formatted_value_col} vs Time (gamma = {gamma_value})')
        ax1.set_xlabel(f'Cumulative {formatted_time_col} (Log)')
        ax1.set_ylabel(f'Cumulative {formatted_value_col} (Log)')
        
        ax2.set_title(f'Percentage Difference from Baseline (nt_opt={baseline_ntopt})')
        ax2.set_xlabel(f'Cumulative {formatted_time_col}')
        ax2.set_ylabel('% Difference')
        
        ax3.set_title(f'Normalized Values (Baseline = 1.0)')
        ax3.set_xlabel(f'Cumulative {formatted_time_col} (Log)')
        ax3.set_ylabel('Normalized Value')
        
        ax4.set_title(f'Absolute Difference from Baseline')
        ax4.set_xlabel(f'Cumulative {formatted_time_col} (Log)')
        ax4.set_ylabel(f'Absolute Difference in {formatted_value_col}')
    
    # Add grids and legends
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    # Add zero reference lines where appropriate
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.95, hspace=0.4, wspace=0.25)
    
    # Save the comprehensive plot
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        safe_time_col = formatted_time_col.replace(' ', '_')
        safe_value_col = formatted_value_col.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"comprehensive_ntopt_analysis_{safe_time_col}_vs_{safe_value_col}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "comprehensive_analysis_loglog.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive LOG-LOG analysis to {save_dir}")
        
        # Save individual subplots
        subplot_names = ['loglog_plot', 'percentage_difference', 'normalized_values', 'absolute_difference']
        for i, (ax, name) in enumerate(zip([ax1, ax2, ax3, ax4], subplot_names)):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Expand the bounding box to include the title
            expanded_extent = extent.expanded(1.3, 1.3)
            # Adjust the expansion to account for the title position
            expanded_extent.y1 += 0.3  # Add space at the top for the title
            subplot_path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(subplot_path, bbox_inches=expanded_extent, dpi=300)
        print(f"Saved individual subplots to {save_dir}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== LOG-LOG DIFFERENCE ANALYSIS SUMMARY for gamma = {gamma_value} ===")
    print(f"Baseline: nt_opt = {baseline_ntopt}")
    
    final_values = {}
    for ntopt_val in sorted_ntopt_values:
        data = ntopt_data[ntopt_val]
        final_value = data['cumulative_values'][-1]
        final_values[ntopt_val] = final_value
        
        if ntopt_val == baseline_ntopt:
            print(f"nt_opt = {ntopt_val}: {final_value:.1f} (BASELINE)")
        else:
            pct_change = (final_value - final_values[baseline_ntopt]) / final_values[baseline_ntopt] * 100
            abs_change = final_value - final_values[baseline_ntopt]
            print(f"nt_opt = {ntopt_val}: {final_value:.1f} ({pct_change:+.2f}%, {abs_change:+.1f})")

def plot_ntopt_comparison_with_differences_linear(gamma_value, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves with comprehensive difference analysis using LINEAR plots for small differences.
    Creates multiple subplots to highlight percentage differences between ntopt values.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma_value}_ntopt_*"
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract nt_opt values and collect data
    ntopt_data = {}
    
    for directory in matching_dirs:
        # Extract nt_opt value from directory name
        match = re.search(r'ntopt_([0-9]+)', os.path.basename(directory))
        if not match:
            continue
            
        ntopt_val = match.group(1)
        
        # Phase 2 network types in order for cumulative plotting
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}_Averaged.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV - use all data points from beginning
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
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in phase3)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                ntopt_data[ntopt_val] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
    
    if not ntopt_data:
        print("No valid data found to plot.")
        return
    
    # Sort nt_opt values for consistent ordering
    sorted_ntopt_values = sorted(ntopt_data.keys(), key=int)
    
    # Use smallest ntopt as baseline
    baseline_ntopt = sorted_ntopt_values[0]
    baseline_data = ntopt_data[baseline_ntopt]
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    # Create comprehensive difference analysis with 4 subplots (LINEAR VERSION)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(sorted_ntopt_values)))
    
    # Collect transition points for vertical lines (using first nt_opt value as reference)
    if sorted_ntopt_values:
        reference_data = ntopt_data[sorted_ntopt_values[0]]
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference nt_opt, recalculate transition points
        ref_ntopt_val = sorted_ntopt_values[0]
        directory = None
        for dir_path in matching_dirs:
            if f"ntopt_{ref_ntopt_val}" in dir_path:
                directory = dir_path
                break
        
        if directory:
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}_Averaged.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Subplot 1: Original LINEAR plot
    for i, ntopt_val in enumerate(sorted_ntopt_values):
        data = ntopt_data[ntopt_val]
        ax1.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 1
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax1.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax1.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax1.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 2: Percentage difference from baseline
    for i, ntopt_val in enumerate(sorted_ntopt_values[1:], 1):  # Skip baseline
        data = ntopt_data[ntopt_val]
        baseline_values_interp = np.interp(data['cumulative_times'],
                                         baseline_data['cumulative_times'],
                                         baseline_data['cumulative_values'])
        pct_diff = (np.array(data['cumulative_values']) - baseline_values_interp) / baseline_values_interp * 100
        
        ax2.plot(data['cumulative_times'], pct_diff,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$ vs $nt_{{opt}} = {baseline_ntopt}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 2
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax2.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax2.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax2.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 3: Normalized values (divide by baseline)
    for i, ntopt_val in enumerate(sorted_ntopt_values):
        data = ntopt_data[ntopt_val]
        if ntopt_val == baseline_ntopt:
            normalized_values = np.ones_like(data['cumulative_values'])  # Baseline = 1
        else:
            baseline_values_interp = np.interp(data['cumulative_times'],
                                             baseline_data['cumulative_times'],
                                             baseline_data['cumulative_values'])
            normalized_values = np.array(data['cumulative_values']) / baseline_values_interp
        
        ax3.plot(data['cumulative_times'], normalized_values,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_val}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 3
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax3.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax3.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax3.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.8)
    
    # Subplot 4: Absolute difference from baseline
    for i, ntopt_val in enumerate(sorted_ntopt_values[1:], 1):  # Skip baseline
        data = ntopt_data[ntopt_val]
        baseline_values_interp = np.interp(data['cumulative_times'],
                                         baseline_data['cumulative_times'],
                                         baseline_data['cumulative_values'])
        abs_diff = np.array(data['cumulative_values']) - baseline_values_interp
        int_val = int(float(ntopt_val) * 0.1)

        print(f"nt_opt value: {ntopt_val}, int_val for labeling: {int_val}")
        ax4.plot(data['cumulative_times'], abs_diff,
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {int_val} - nt_{{opt}} = {baseline_ntopt}$', alpha=0.8)
    
    # Add vertical lines and labels for network transitions to subplot 4
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = ax4.get_ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            ax4.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            ax4.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=16, alpha=0.8)
    
    # Set titles and labels
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        ax1.set_title(f'Linear: Cumulative {formatted_value_col} vs Time (gamma = {gamma_value})')
        ax1.set_xlabel(f'Cumulative {formatted_time_col}')
        ax1.set_ylabel(f'Cumulative {formatted_value_col}')
        
        ax2.set_title(f'Percentage Difference from Baseline (nt_opt={baseline_ntopt})')
        ax2.set_xlabel(f'Cumulative {formatted_time_col}')
        ax2.set_ylabel('% Difference')
        
        ax3.set_title(f'Normalized Values (Baseline = 1.0)')
        ax3.set_xlabel(f'Cumulative {formatted_time_col}')
        ax3.set_ylabel('Normalized Value')
        
        ax4.set_title(f'Absolute Difference from Baseline')
        ax4.set_xlabel(f'Cumulative {formatted_time_col}')
        ax4.set_ylabel(f'Absolute Difference in {formatted_value_col}')
    
    # Add grids and legends
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    # Add zero reference lines where appropriate
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    
    # Save the comprehensive plot
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        safe_time_col = formatted_time_col.replace(' ', '_')
        safe_value_col = formatted_value_col.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"comprehensive_ntopt_analysis_linear_{safe_time_col}_vs_{safe_value_col}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "comprehensive_analysis_linear.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive LINEAR analysis to {save_dir}")
        
        # Save individual subplots
        subplot_names = ['linear_plot', 'percentage_difference', 'normalized_values', 'absolute_difference']
        for i, (ax, name) in enumerate(zip([ax1, ax2, ax3, ax4], subplot_names)):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Expand the bounding box to include the title
            expanded_extent = extent.expanded(1.3, 1.3)
            # Adjust the expansion to account for the title position
            expanded_extent.y1 += 0.1  # Add space at the top for the title
            subplot_path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(subplot_path, bbox_inches=expanded_extent, dpi=300)
        print(f"Saved individual subplots to {save_dir}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== LINEAR DIFFERENCE ANALYSIS SUMMARY for gamma = {gamma_value} ===")
    print(f"Baseline: nt_opt = {baseline_ntopt}")
    
    final_values = {}
    for ntopt_val in sorted_ntopt_values:
        data = ntopt_data[ntopt_val]
        final_value = data['cumulative_values'][-1]
        final_values[ntopt_val] = final_value
        
        if ntopt_val == baseline_ntopt:
            print(f"nt_opt = {ntopt_val}: {final_value:.1f} (BASELINE)")
        else:
            pct_change = (final_value - final_values[baseline_ntopt]) / final_values[baseline_ntopt] * 100
            abs_change = final_value - final_values[baseline_ntopt]
            print(f"nt_opt = {ntopt_val}: {final_value:.1f} ({pct_change:+.2f}%, {abs_change:+.1f})")

def plot_specific_ntopt_comparison(gamma_value, ntopt_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for specific nt_opt values with points at regular timestep intervals.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        ntopt_values (list): List of nt_opt values to plot (e.g., ["0", "10", "600"])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    
    # Collect data for specified nt_opt values
    ntopt_data = {}
    
    # Sort nt_opt values for consistent ordering
    sorted_ntopt_values = sorted([int(n) for n in ntopt_values])
    
    for ntopt_val in sorted_ntopt_values:
        ntopt_str = str(ntopt_val)
        directory_name = f"gamma_{gamma_value}_ntopt_{ntopt_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if not os.path.exists(directory):
            print(f"Warning: No directory found for gamma = {gamma_value}, nt_opt = {ntopt_str}")
            continue
        
        # Phase 2 network types in order for cumulative plotting
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}_Averaged.csv"
            csv_path = os.path.join(directory, csv_filename)
            
            if os.path.exists(csv_path):
                if i == 0:  # First CSV - use all data points from beginning
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
                        print(f"nt_opt={ntopt_str}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - collect all points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # For subsequent CSVs, we want the final value only (like in phase3)
                        if time_vals and data_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            final_data = data_vals[-1]  # Take the last data value
                            cumulative_time += final_time
                            cumulative_value += final_data
                            all_cumulative_times.append(cumulative_time)
                            all_cumulative_values.append(cumulative_value)
                            print(f"nt_opt={ntopt_str}, {network_type}: time={final_time}, value={final_data}, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                ntopt_data[ntopt_str] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"nt_opt={ntopt_str}: Created {len(equidistant_times)} equidistant points")
                print(f"nt_opt={ntopt_str}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not ntopt_data:
        print("No valid data found to plot.")
        return
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(ntopt_data)))
    
    # Collect transition points for vertical lines (using first nt_opt value as reference)
    if ntopt_data:
        sorted_items = sorted(ntopt_data.items(), key=lambda x: int(x[0]))
        reference_ntopt_str = sorted_items[0][0]
        network_types = ['AM Base', 'AM 2', 'AM 3', 'PM Base']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference nt_opt, recalculate transition points
        directory_name = f"gamma_{gamma_value}_ntopt_{reference_ntopt_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if os.path.exists(directory):
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}_Averaged.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                if os.path.exists(csv_path):
                    if i == 0:  # First CSV
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
                    else:  # Subsequent CSVs
                        time_vals, _, _ = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                        if time_vals is not None and time_vals:
                            final_time = time_vals[-1]  # Take the last time value
                            cumulative_time += final_time
                            transition_points.append((cumulative_time, network_type))
    
    # Plot each nt_opt value as a separate line
    for i, (ntopt_str, data) in enumerate(sorted(ntopt_data.items(), key=lambda x: int(x[0]))):
        plt.loglog(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_str}$', alpha=0.8)
    
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
    for data in ntopt_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        plt.xlabel(f'Cumulative {formatted_time_col}')
        plt.ylabel(f'Cumulative {formatted_value_col}')
        plt.title(f'Cumulative {formatted_value_col} vs Cumulative {formatted_time_col} - Selected nt_opt Values (gamma = {gamma_value})')
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        plt.title(f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index} - Selected nt_opt Values (gamma = {gamma_value})')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        safe_time_col = formatted_time_col.replace(' ', '_')
        safe_value_col = formatted_value_col.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"specific_ntopt_comparison_detailed_cumulative_{safe_time_col}_vs_cumulative_{safe_value_col}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {len(ntopt_data)} selected nt_opt values")
    if ntopt_data:
        print(f"Plotted nt_opt values: {list(ntopt_data.keys())}")

def plot_single_network_ntopt_comparison(gamma_value, network_type, ntopt_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across different nt_opt values.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        network_type (str): Network type (e.g., "PM Base", "AM Base", etc.)
        ntopt_values (list): List of nt_opt values to plot (e.g., ["0", "10", "600"])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    
    # Collect data for specified nt_opt values
    ntopt_data = {}
    
    # Sort nt_opt values for consistent ordering
    sorted_ntopt_values = sorted([int(n) for n in ntopt_values])
    
    for ntopt_val in sorted_ntopt_values:
        ntopt_str = str(ntopt_val)
        directory_name = f"gamma_{gamma_value}_ntopt_{ntopt_str}"
        directory = os.path.join(base_dir, directory_name)
        
        if not os.path.exists(directory):
            print(f"Warning: No directory found for gamma = {gamma_value}, nt_opt = {ntopt_str}")
            continue
        
        csv_filename = f"time_series_data_{network_type}_Averaged.csv"
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
                        ntopt_data[ntopt_str] = {
                            'times': equidistant_times,
                            'values': equidistant_values,
                            'column_names': cols
                        }
                        print(f"nt_opt={ntopt_str}, {network_type}: Created {len(equidistant_times)} equidistant points")
                        print(f"nt_opt={ntopt_str}: First 10 times: {equidistant_times[:10]}")
                else:
                    # Use all data points for smaller intervals
                    ntopt_data[ntopt_str] = {
                        'times': time_vals,
                        'values': data_vals,
                        'column_names': cols
                    }
                    print(f"nt_opt={ntopt_str}, {network_type}: Using all {len(time_vals)} data points")
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    if not ntopt_data:
        print("No valid data found to plot.")
        return
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': GLOBAL_FONT_SIZE,
        'axes.titlesize': GLOBAL_TITLE_SIZE,
        'axes.labelsize': GLOBAL_LABEL_SIZE,
        'xtick.labelsize': GLOBAL_TICK_SIZE,
        'ytick.labelsize': GLOBAL_TICK_SIZE,
        'legend.fontsize': GLOBAL_LEGEND_SIZE
    })
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(ntopt_data)))
    
    # Plot each nt_opt value as a separate line
    for i, (ntopt_str, data) in enumerate(sorted(ntopt_data.items(), key=lambda x: int(x[0]))):
        plt.loglog(data['times'], data['values'],
                'o-', linewidth=2, markersize=3, color=colors[i],
                label=f'$nt_{{opt}} = {ntopt_str}$', alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in ntopt_data.values():
        if data['column_names']:
            column_names = data['column_names']
            break
    
    # Set labels and title
    if column_names:
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        plt.xlabel(f'{formatted_time_col}')
        plt.ylabel(f'{formatted_value_col}')
        plt.title(f'{formatted_value_col} vs {formatted_time_col} for {network_type} - Different nt_opt Values (gamma = {gamma_value})')
    else:
        plt.xlabel(f'Column {time_col_index}')
        plt.ylabel(f'Column {value_col_index}')
        plt.title(f'Column {value_col_index} vs Column {time_col_index} for {network_type} - Different nt_opt Values (gamma = {gamma_value})')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        safe_network_name = network_type.replace(' ', '_')
        formatted_time_col = format_column_name(column_names[time_col_index])
        formatted_value_col = format_column_name(column_names[value_col_index])
        safe_time_col = formatted_time_col.replace(' ', '_')
        safe_value_col = formatted_value_col.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma_{gamma_value}_plots", f"single_network_{safe_network_name}_ntopt_comparison_{safe_time_col}_vs_{safe_value_col}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {network_type} with {len(ntopt_data)} nt_opt values")
    if ntopt_data:
        print(f"Plotted nt_opt values: {list(ntopt_data.keys())}")

def plot_all_ntopt_single_network(gamma_value, network_type, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across ALL available nt_opt values.
    
    Args:
        gamma_value (str): Gamma value (e.g., "0.0375")
        network_type (str): Network type (e.g., "PM Base", "AM Base", etc.)
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    pattern = f"gamma_{gamma_value}_ntopt_*"
    
    # Find all matching directories for this gamma value
    search_pattern = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {search_pattern}")
        return
    
    # Extract nt_opt values from directory names
    ntopt_values = []
    for directory in matching_dirs:
        match = re.search(r'ntopt_([0-9]+)', os.path.basename(directory))
        if match:
            # Check if the specific network CSV exists in this directory
            csv_filename = f"time_series_data_{network_type}_Averaged.csv"
            csv_path = os.path.join(directory, csv_filename)
            if os.path.exists(csv_path):
                ntopt_values.append(match.group(1))
    
    if not ntopt_values:
        print(f"No valid nt_opt directories with {network_type} data found for gamma = {gamma_value}")
        return
    
    print(f"Found {len(ntopt_values)} nt_opt values for {network_type}: {sorted(ntopt_values, key=int)}")
    
    # Use the existing single network plotting function
    plot_single_network_ntopt_comparison(gamma_value, network_type, ntopt_values, time_col_index, value_col_index, timestep_interval)

def main():
    """
    Main function to handle command line arguments or interactive input.
    """
    if len(sys.argv) >= 2:
        if sys.argv[1] == "specific":
            # Specific nt_opt values mode: python script.py specific gamma_value ntopt_1,ntopt_2,ntopt_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 4:
                print("Error: Please provide gamma value and nt_opt values. Usage: python script.py specific 0.0375 0,10,600 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma_value = sys.argv[2]
            ntopt_str = sys.argv[3]
            ntopt_values = [n.strip() for n in ntopt_str.split(',')]
            time_col_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            value_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            timestep_interval = int(float(sys.argv[6])) if len(sys.argv) > 6 else 1000
            
            plot_specific_ntopt_comparison(gamma_value, ntopt_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single":
            # Single network mode: python script.py single gamma_value "network_type" ntopt_1,ntopt_2,ntopt_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 5:
                print("Error: Please provide gamma value, network type, and nt_opt values. Usage: python script.py single 0.0375 \"AM Base\" 0,10,600 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma_value = sys.argv[2]
            network_type = sys.argv[3]
            ntopt_str = sys.argv[4]
            ntopt_values = [n.strip() for n in ntopt_str.split(',')]
            time_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            value_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 1
            timestep_interval = int(float(sys.argv[7])) if len(sys.argv) > 7 else 1000
            
            plot_single_network_ntopt_comparison(gamma_value, network_type, ntopt_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single_all":
            # Single network ALL nt_opt mode: python script.py single_all gamma_value "network_type" [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 4:
                print("Error: Please provide gamma value and network type. Usage: python script.py single_all 0.0375 \"AM Base\" [time_col] [value_col] [timestep_interval]")
                return
            
            gamma_value = sys.argv[2]
            network_type = sys.argv[3]
            time_col_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            value_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            timestep_interval = int(float(sys.argv[6])) if len(sys.argv) > 6 else 1000
            
            plot_all_ntopt_single_network(gamma_value, network_type, time_col_index, value_col_index, timestep_interval)
        else:
            # nt_opt comparison mode
            gamma_value = sys.argv[1]
            time_col_index = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
            value_col_index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
            timestep_interval = int(float(sys.argv[4])) if len(sys.argv) > 4 else 1000
            plot_ntopt_comparison(gamma_value, time_col_index, value_col_index, timestep_interval)
    else:
        # Interactive mode
        print("Choose mode:")
        print("1. Plot comparison for all nt_opt values for a gamma")
        print("2. Plot comparison for specific nt_opt values for a gamma")
        print("3. Plot comparison for a single network type across specific nt_opt values")
        print("4. Plot comparison for a single network type across ALL nt_opt values")
        print("5. Comprehensive difference analysis - LOG-LOG (4-subplot view for small differences)")
        print("6. Comprehensive difference analysis - LINEAR (4-subplot view for small differences)")
        
        choice = input("Enter choice (1, 2, 3, 4, 5, or 6): ").strip()
        
        if choice == "1":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_ntopt_comparison(gamma_value, time_col_index, value_col_index, timestep_interval)
        elif choice == "2":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            ntopt_str = input("Enter nt_opt values separated by commas (e.g., 0, 10, 600): ").strip()
            ntopt_values = [n.strip() for n in ntopt_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_specific_ntopt_comparison(gamma_value, ntopt_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "3":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            network_type = input("Enter network type (e.g., 'AM Base', 'PM Base', 'AM 2', etc.): ").strip()
            ntopt_str = input("Enter nt_opt values separated by commas (e.g., 0, 10, 600): ").strip()
            ntopt_values = [n.strip() for n in ntopt_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_single_network_ntopt_comparison(gamma_value, network_type, ntopt_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "4":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            network_type = input("Enter network type (e.g., 'AM Base', 'PM Base', 'AM 2', etc.): ").strip()
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_all_ntopt_single_network(gamma_value, network_type, time_col_index, value_col_index, timestep_interval)
        elif choice == "5":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_ntopt_comparison_with_differences(gamma_value, time_col_index, value_col_index, timestep_interval)
        elif choice == "6":
            gamma_value = input("Enter gamma value (e.g., 0.0375): ").strip()
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_ntopt_comparison_with_differences_linear(gamma_value, time_col_index, value_col_index, timestep_interval)
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

if __name__ == "__main__":
    main()
