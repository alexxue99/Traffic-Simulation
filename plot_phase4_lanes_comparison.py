import csv
import matplotlib.pyplot as plt
import sys
import os
import glob
import re
import numpy as np

## Global Variables
BASE_DIR = "phase4_experiments"  # which parent folder to look for results
DT = 0.1  # Time step for scaling values

# Global font size parameters - Modify these to change text sizes globally
GLOBAL_FONT_SIZE = 32
GLOBAL_TITLE_SIZE = 27
GLOBAL_LABEL_SIZE = 27
GLOBAL_TICK_SIZE = 20
GLOBAL_LEGEND_SIZE = 20
TRANSITION_FONT_SIZE = 25
LINE_WIDTH = 2

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
    
    # Create equidistant time points starting from the actual minimum time
    # up to the maximum time, at exact multiples of timestep_interval
    equidistant_times = []
    current_time = min_time
    while current_time <= max_time:
        equidistant_times.append(current_time)
        current_time += timestep_interval
    
    # Include the actual max_time as the last point to ensure complete coverage
    if equidistant_times and max_time > equidistant_times[-1]:
        equidistant_times.append(max_time)
    
    # Interpolate values at equidistant time points
    if equidistant_times:
        equidistant_values = np.interp(equidistant_times, sorted_times, sorted_values)
        return equidistant_times, equidistant_values.tolist()
    else:
        return [], []

def plot_lanes_comparison(gamma1_value, gamma2_value, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for different lane numbers with points at regular timestep intervals.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0375")
        gamma2_value (str): Gamma2 value (e.g., "0.1000")
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_value}"
    directory = os.path.join(base_dir, directory_name)
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return
    
    # Find all lane subdirectories (e.g., "2_lanes", "3_lanes")
    lane_pattern = os.path.join(directory, "*_lanes")
    lane_dirs = glob.glob(lane_pattern)
    
    if not lane_dirs:
        print(f"No lane directories found matching pattern: {lane_pattern}")
        # Also check for data directly in the main directory (no lane subdir)
        lane_dirs = [directory]
    
    # Extract lane numbers and sort them
    lane_data = {}
    
    for lane_dir in lane_dirs:
        # Extract lane number from directory name
        match = re.search(r'(\d+)_lanes', os.path.basename(lane_dir))
        if match:
            lane_num = int(match.group(1))
        else:
            lane_num = 1  # Default if no lane subdirectory (data directly in main dir)
        
        # Network types in order for cumulative plotting - Only PM networks 2-4
        network_types = ['PM 2', 'PM 3', 'PM 4']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(lane_dir, csv_filename)
            
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
                        print(f"Lanes={lane_num}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - add previous CSV's final value to ALL points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # Add all points from this CSV, offset by the previous CSV's final cumulative value
                        for time_val, data_val in zip(time_vals, data_vals):
                            all_cumulative_times.append(cumulative_time + time_val)
                            all_cumulative_values.append(cumulative_value + data_val)
                        
                        # Update the cumulative offset to the final value of this CSV
                        cumulative_time += time_vals[-1]
                        cumulative_value += data_vals[-1]
                        print(f"Lanes={lane_num}, {network_type}: Added {len(time_vals)} points, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                lane_data[lane_num] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"Lanes={lane_num}: Created {len(equidistant_times)} equidistant points")
                print(f"Lanes={lane_num}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not lane_data:
        print("No valid data found to plot.")
        return
    
    # Sort lane numbers for consistent ordering
    sorted_lane_values = sorted(lane_data.keys())
    
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
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(sorted_lane_values)))
    
    # Collect transition points for vertical lines (using first lane value as reference)
    if sorted_lane_values:
        reference_lane = sorted_lane_values[0]
        network_types = ['PM 2', 'PM 3', 'PM 4']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference lane, recalculate transition points
        ref_lane_dir = None
        if reference_lane == 1:
            ref_lane_dir = directory
        else:
            ref_lane_dir = os.path.join(directory, f"{reference_lane}_lanes")
        
        if ref_lane_dir and os.path.exists(ref_lane_dir):
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}.csv"
                csv_path = os.path.join(ref_lane_dir, csv_filename)
                
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
    
    # Plot each lane number as a separate line
    for i, lane_num in enumerate(sorted_lane_values):
        data = lane_data[lane_num]
        plt.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=LINE_WIDTH, markersize=3, color=colors[i],
                label=f'{lane_num} Lanes', alpha=0.8)
    
    # Add vertical lines and labels for network transitions
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = plt.ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            plt.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=TRANSITION_FONT_SIZE, alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in lane_data.values():
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
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'{y_label} vs {x_label}'
        title_line2 = f'for Different Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index}'
        title_line2 = f'for Different Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right')
    
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_gamma2_{gamma2_value}_plots", f"lanes_comparison_cumulative_{column_names[time_col_index]}_vs_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for $\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value}")
    print(f"Found {len(sorted_lane_values)} lane configurations: {sorted_lane_values}")

def plot_specific_lanes_comparison(gamma1_value, gamma2_value, lane_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot cumulative curves for specific lane numbers with points at regular timestep intervals.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0375")
        gamma2_value (str): Gamma2 value (e.g., "0.1000")
        lane_values (list): List of lane numbers to plot (e.g., [2, 3, 4])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_value}"
    directory = os.path.join(base_dir, directory_name)
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return
    
    # Collect data for specified lane values
    lane_data = {}
    
    # Sort lane values for consistent ordering
    sorted_lane_values = sorted([int(l) for l in lane_values])
    
    for lane_num in sorted_lane_values:
        lane_dir = os.path.join(directory, f"{lane_num}_lanes")
        
        if not os.path.exists(lane_dir):
            print(f"Warning: Lane directory not found for lane = {lane_num}: {lane_dir}")
            continue
        
        # Network types in order for cumulative plotting - Only PM networks 2-4
        network_types = ['PM 2', 'PM 3', 'PM 4']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(lane_dir, csv_filename)
            
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
                        print(f"Lanes={lane_num}, {network_type}: Added {len(time_vals)} points, final time={cumulative_time}, final value={cumulative_value}")
                else:  # Subsequent CSVs - add previous CSV's final value to ALL points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # Add all points from this CSV, offset by the previous CSV's final cumulative value
                        for time_val, data_val in zip(time_vals, data_vals):
                            all_cumulative_times.append(cumulative_time + time_val)
                            all_cumulative_values.append(cumulative_value + data_val)
                        
                        # Update the cumulative offset to the final value of this CSV
                        cumulative_time += time_vals[-1]
                        cumulative_value += data_vals[-1]
                        print(f"Lanes={lane_num}, {network_type}: Added {len(time_vals)} points, cumulative_time={cumulative_time}, cumulative_value={cumulative_value}")
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                lane_data[lane_num] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
                print(f"Lanes={lane_num}: Created {len(equidistant_times)} equidistant points")
                print(f"Lanes={lane_num}: First 10 equidistant times: {equidistant_times[:10]}")
    
    if not lane_data:
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
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(lane_data)))
    
    # Collect transition points for vertical lines (using first lane value as reference)
    if lane_data:
        sorted_items = sorted(lane_data.items(), key=lambda x: x[0])
        reference_lane = sorted_items[0][0]
        network_types = ['PM 2', 'PM 3', 'PM 4']
        
        # Calculate transition points based on the reference data
        transition_points = []
        cumulative_time = 0
        
        # For the reference lane, recalculate transition points
        ref_lane_dir = os.path.join(directory, f"{reference_lane}_lanes")
        
        if os.path.exists(ref_lane_dir):
            for i, network_type in enumerate(network_types):
                csv_filename = f"time_series_data_{network_type}.csv"
                csv_path = os.path.join(ref_lane_dir, csv_filename)
                
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
    
    # Plot each lane number as a separate line
    for i, (lane_num, data) in enumerate(sorted(lane_data.items())):
        plt.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=LINE_WIDTH, markersize=3, color=colors[i],
                label=f'{lane_num} Lanes', alpha=0.8)
    
    # Add vertical lines and labels for network transitions
    if 'transition_points' in locals() and transition_points:
        y_min, y_max = plt.ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):  # Exclude the last point
            # Add vertical dotted line
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=2)

            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            plt.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=TRANSITION_FONT_SIZE, alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in lane_data.values():
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
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'{y_label} vs {x_label}'
        title_line2 = f'- Selected Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index}')
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'Cumulative Column {value_col_index} vs Cumulative Column {time_col_index}'
        title_line2 = f'- Selected Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right')
    
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_gamma2_{gamma2_value}_plots", f"specific_lanes_comparison_cumulative_{column_names[time_col_index]}_vs_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {len(lane_data)} selected lane values")
    if lane_data:
        print(f"Plotted lane values: {list(lane_data.keys())}")

def plot_single_network_lanes_comparison(gamma1_value, gamma2_value, network_type, lane_values, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across different lane numbers.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0375")
        gamma2_value (str): Gamma2 value (e.g., "0.1000")
        network_type (str): Network type (e.g., "PM 5", "AM Base", etc.)
        lane_values (list): List of lane numbers to plot (e.g., [2, 3, 4])
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_value}"
    directory = os.path.join(base_dir, directory_name)
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return
    
    # Collect data for specified lane values
    lane_data = {}
    
    # Sort lane values for consistent ordering
    sorted_lane_values = sorted([int(l) for l in lane_values])
    
    for lane_num in sorted_lane_values:
        lane_dir = os.path.join(directory, f"{lane_num}_lanes")
        
        if not os.path.exists(lane_dir):
            print(f"Warning: Lane directory not found for lane = {lane_num}: {lane_dir}")
            continue
        
        csv_filename = f"time_series_data_{network_type}.csv"
        csv_path = os.path.join(lane_dir, csv_filename)
        
        if os.path.exists(csv_path):
            # Read all data points for this network
            time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
            if time_vals is not None and data_vals is not None:
                # Create equidistant sampling if needed
                if timestep_interval > 100:  # Only do equidistant sampling for larger intervals
                    equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                        time_vals, data_vals, timestep_interval)
                    if equidistant_times:
                        lane_data[lane_num] = {
                            'times': equidistant_times,
                            'values': equidistant_values,
                            'column_names': cols
                        }
                        print(f"Lanes={lane_num}, {network_type}: Created {len(equidistant_times)} equidistant points")
                        print(f"Lanes={lane_num}: First 10 times: {equidistant_times[:10]}")
                else:
                    # Use all data points for smaller intervals
                    lane_data[lane_num] = {
                        'times': time_vals,
                        'values': data_vals,
                        'column_names': cols
                    }
                    print(f"Lanes={lane_num}, {network_type}: Using all {len(time_vals)} data points")
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    if not lane_data:
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
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(lane_data)))
    
    # Plot each lane number as a separate line
    for i, (lane_num, data) in enumerate(sorted(lane_data.items())):
        plt.plot(data['times'], data['values'],
                'o-', linewidth=LINE_WIDTH, markersize=3, color=colors[i],
                label=f'{lane_num} Lanes', alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in lane_data.values():
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
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'{y_label} vs {x_label} for {network_type}'
        title_line2 = f'- Different Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    else:
        plt.xlabel(f'Column {time_col_index}')
        plt.ylabel(f'Column {value_col_index}')
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'Column {value_col_index} vs Column {time_col_index} for {network_type}'
        title_line2 = f'- Different Lane Numbers ($\\gamma_1$ = {gamma1_value}, $\\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right')
    
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        safe_network_name = network_type.replace(' ', '_')
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_gamma2_{gamma2_value}_plots", f"single_network_{safe_network_name}_lanes_comparison_{column_names[time_col_index]}_vs_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)   # make sure folder exists

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted comparison for {network_type} with {len(lane_data)} lane values")
    if lane_data:
        print(f"Plotted lane values: {list(lane_data.keys())}")

def plot_all_lanes_single_network(gamma1_value, gamma2_value, network_type, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot curves for a single network type across ALL available lane values.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0375")
        gamma2_value (str): Gamma2 value (e.g., "0.1000")
        network_type (str): Network type (e.g., "PM 5", "AM Base", etc.)
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_value}"
    directory = os.path.join(base_dir, directory_name)
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return
    
    # Find all lane subdirectories
    lane_pattern = os.path.join(directory, "*_lanes")
    lane_dirs = glob.glob(lane_pattern)
    
    # Extract lane numbers from directory names
    lane_values = []
    for lane_dir in lane_dirs:
        match = re.search(r'(\d+)_lanes', os.path.basename(lane_dir))
        if match:
            # Check if the specific network CSV exists in this directory
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(lane_dir, csv_filename)
            if os.path.exists(csv_path):
                lane_values.append(match.group(1))
    
    if not lane_values:
        print(f"No valid lane directories with {network_type} data found for gamma1 = {gamma1_value}, gamma2 = {gamma2_value}")
        return
    
    print(f"Found {len(lane_values)} lane values for {network_type}: {sorted(lane_values, key=int)}")
    
    # Use the existing single network plotting function
    plot_single_network_lanes_comparison(gamma1_value, gamma2_value, network_type, lane_values, time_col_index, value_col_index, timestep_interval)

def plot_lanes_difference(gamma1_value, gamma2_value, baseline_lane=2, time_col_index=0, value_col_index=1, timestep_interval=1000):
    """
    Plot the difference in cumulative values between lane configurations and a baseline lane.
    
    Args:
        gamma1_value (str): Gamma1 value (e.g., "0.0375")
        gamma2_value (str): Gamma2 value (e.g., "0.1000")
        baseline_lane (int): Baseline lane number to subtract from others (default: 2)
        time_col_index (int): Index of time column (default: 0)
        value_col_index (int): Index of value column (default: 1 for weighted_integrated_cars)
        timestep_interval (float): Interval between timesteps to extract (default: 1000)
    """
    base_dir = BASE_DIR
    directory_name = f"gamma_{gamma1_value}_gamma2_{gamma2_value}"
    directory = os.path.join(base_dir, directory_name)
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return
    
    # Find all lane subdirectories
    lane_pattern = os.path.join(directory, "*_lanes")
    lane_dirs = glob.glob(lane_pattern)
    
    if not lane_dirs:
        print(f"No lane directories found matching pattern: {lane_pattern}")
        return
    
    # Extract lane numbers and sort them
    lane_data = {}
    
    for lane_dir in lane_dirs:
        # Extract lane number from directory name
        match = re.search(r'(\d+)_lanes', os.path.basename(lane_dir))
        if match:
            lane_num = int(match.group(1))
        else:
            continue
        
        # Network types in order for cumulative plotting - Only PM networks 2-4
        network_types = ['PM 2', 'PM 3', 'PM 4']
        
        # Collect ALL cumulative time-value pairs first
        all_cumulative_times = []
        all_cumulative_values = []
        cumulative_time = 0
        cumulative_value = 0
        column_names = None
        
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(lane_dir, csv_filename)
            
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
                else:  # Subsequent CSVs - add previous CSV's final value to ALL points
                    time_vals, data_vals, cols = read_csv_by_timestep(csv_path, time_col_index, value_col_index, timestep_interval)
                    if time_vals is not None and data_vals is not None:
                        if column_names is None:
                            column_names = cols
                        
                        # Add all points from this CSV, offset by the previous CSV's final cumulative value
                        for time_val, data_val in zip(time_vals, data_vals):
                            all_cumulative_times.append(cumulative_time + time_val)
                            all_cumulative_values.append(cumulative_value + data_val)
                        
                        # Update the cumulative offset to the final value of this CSV
                        cumulative_time += time_vals[-1]
                        cumulative_value += data_vals[-1]
        
        # Now create equidistant sampling from the complete cumulative data
        if all_cumulative_times:
            equidistant_times, equidistant_values = create_equidistant_cumulative_data(
                all_cumulative_times, all_cumulative_values, timestep_interval)
            
            if equidistant_times:
                lane_data[lane_num] = {
                    'cumulative_times': equidistant_times,
                    'cumulative_values': equidistant_values,
                    'column_names': column_names
                }
    
    # Check if baseline lane exists
    if baseline_lane not in lane_data:
        print(f"Warning: Baseline lane {baseline_lane} not found in data. Available lanes: {sorted(lane_data.keys())}")
        return
    
    # Get baseline data
    baseline_data = lane_data[baseline_lane]
    
    # Calculate differences for all other lanes
    difference_data = {}
    for lane_num, data in lane_data.items():
        if lane_num == baseline_lane:
            continue
        
        # Interpolate baseline values at the same time points as the comparison lane
        baseline_interp = np.interp(data['cumulative_times'], baseline_data['cumulative_times'], baseline_data['cumulative_values'])
        
        # Calculate difference (comparison - baseline)
        difference_values = [comp - base for comp, base in zip(data['cumulative_values'], baseline_interp)]
        
        difference_data[lane_num] = {
            'cumulative_times': data['cumulative_times'],
            'cumulative_values': difference_values,
            'column_names': data['column_names']
        }
        print(f"Difference ({lane_num} Lanes  - {baseline_lane} Lanes ): Final difference = {difference_values[-1]:.2f}")
    
    if not difference_data:
        print("No difference data to plot (only baseline lane found).")
        return
    
    # Sort lane numbers for consistent ordering
    sorted_lane_values = sorted(difference_data.keys())
    
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
    
    colors = plt.colormaps['tab10'](np.linspace(0, 1, 3))
    
    # Plot each difference as a separate line
    for i, lane_num in enumerate(sorted_lane_values):
        data = difference_data[lane_num]
        plt.plot(data['cumulative_times'], data['cumulative_values'],
                'o-', linewidth=LINE_WIDTH, markersize=3, color=colors[i+1],
                label=f'{lane_num} Lanes  - {baseline_lane} Lanes ', alpha=0.8)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Collect transition points for vertical lines (using baseline lane data)
    network_types = ['PM 2', 'PM 3', 'PM 4']
    transition_points = []
    cumulative_time = 0
    
    baseline_lane_dir = os.path.join(directory, f"{baseline_lane}_lanes")
    if os.path.exists(baseline_lane_dir):
        for i, network_type in enumerate(network_types):
            csv_filename = f"time_series_data_{network_type}.csv"
            csv_path = os.path.join(baseline_lane_dir, csv_filename)
            
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
                        final_time = time_vals[-1]
                        cumulative_time += final_time
                        transition_points.append((cumulative_time, network_type))
    
    # Add vertical lines and labels for network transitions
    if transition_points:
        y_min, y_max = plt.ylim()
        for i, (transition_time, network_name) in enumerate(transition_points[:-1]):
            # Add vertical dotted line
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            
            # Add label at the top of the plot
            label_text = f"{network_name} → {transition_points[i+1][1] if i+1 < len(transition_points) else 'End'}"
            plt.text(transition_time, y_max * 0.95, label_text,
                    rotation=90, ha='right', va='top', fontsize=TRANSITION_FONT_SIZE, alpha=0.8)
    
    # Get column names for labeling
    column_names = None
    for data in difference_data.values():
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
        y_label = f'Cumulative {format_label(column_names[value_col_index])} Difference'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'{y_label} vs {x_label}'
        title_line2 = f'(Lanes - {baseline_lane} Lanes Baseline) ($\gamma_1$ = {gamma1_value}, $\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    else:
        plt.xlabel(f'Cumulative Column {time_col_index}')
        plt.ylabel(f'Cumulative Column {value_col_index} Difference')
        # Split title across two lines and use suptitle to center over plot+legend
        title_line1 = f'Cumulative Difference vs Cumulative Column {time_col_index}'
        title_line2 = f'(Lanes - {baseline_lane} Lanes Baseline) ($\gamma_1$ = {gamma1_value}, $\gamma_2$ = {gamma2_value})'
        plt.gcf().suptitle(f'{title_line1}\n{title_line2}', fontsize=GLOBAL_TITLE_SIZE, y=0.98)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right')
    
    plt.tight_layout()

    # Save the plot
    if column_names:
        # Build directory first
        save_dir = os.path.join(base_dir, f"gamma1_{gamma1_value}_gamma2_{gamma2_value}_plots", f"lanes_difference_vs_baseline_{baseline_lane}_{column_names[time_col_index]}_vs_{column_names[value_col_index]}")
        os.makedirs(save_dir, exist_ok=True)

        # Build full file path inside that folder
        save_path = os.path.join(save_dir, "plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Saved figure to {save_dir}")

    # Show the plot
    plt.show()
    
    print(f"\nSuccessfully plotted difference from baseline (Lanes {baseline_lane}) for $\gamma_1$ = {gamma1_value}, $\gamma_2$ = {gamma2_value}")
    print(f"Plotted differences for lanes: {sorted_lane_values}")

def main():
    """
    Main function to handle command line arguments or interactive input.
    """
    if len(sys.argv) >= 2:
        if sys.argv[1] == "specific":
            # Specific lane values mode: python script.py specific gamma1_value gamma2_value lane_1,lane_2,lane_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 5:
                print("Error: Please provide gamma1 value, gamma2 value, and lane values. Usage: python script.py specific 0.0375 0.1000 2,3,4 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            gamma2_value = sys.argv[3]
            lane_str = sys.argv[4]
            lane_values = [l.strip() for l in lane_str.split(',')]
            time_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            value_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 1
            timestep_interval = int(float(sys.argv[7])) if len(sys.argv) > 7 else 1000
            
            plot_specific_lanes_comparison(gamma1_value, gamma2_value, lane_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single":
            # Single network mode: python script.py single gamma1_value gamma2_value "network_type" lane_1,lane_2,lane_3 [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 6:
                print("Error: Please provide gamma1 value, gamma2 value, network type, and lane values. Usage: python script.py single 0.0375 0.1000 \"PM 5\" 2,3,4 [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            gamma2_value = sys.argv[3]
            network_type = sys.argv[4]
            lane_str = sys.argv[5]
            lane_values = [l.strip() for l in lane_str.split(',')]
            time_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
            value_col_index = int(sys.argv[7]) if len(sys.argv) > 7 else 1
            timestep_interval = int(float(sys.argv[8])) if len(sys.argv) > 8 else 1000
            
            plot_single_network_lanes_comparison(gamma1_value, gamma2_value, network_type, lane_values, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "single_all":
            # Single network ALL lanes mode: python script.py single_all gamma1_value gamma2_value "network_type" [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 5:
                print("Error: Please provide gamma1 value, gamma2 value, and network type. Usage: python script.py single_all 0.0375 0.1000 \"PM 5\" [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            gamma2_value = sys.argv[3]
            network_type = sys.argv[4]
            time_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            value_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 1
            timestep_interval = int(float(sys.argv[7])) if len(sys.argv) > 7 else 1000
            
            plot_all_lanes_single_network(gamma1_value, gamma2_value, network_type, time_col_index, value_col_index, timestep_interval)
        elif sys.argv[1] == "diff":
            # Difference mode: python script.py diff gamma1_value gamma2_value [baseline_lane] [time_col] [value_col] [timestep_interval]
            if len(sys.argv) < 3:
                print("Error: Please provide gamma1 and gamma2 values. Usage: python script.py diff 0.0375 0.1000 [baseline_lane] [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[2]
            gamma2_value = sys.argv[3]
            baseline_lane = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            time_col_index = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            value_col_index = int(sys.argv[6]) if len(sys.argv) > 6 else 1
            timestep_interval = int(float(sys.argv[7])) if len(sys.argv) > 7 else 1000
            
            plot_lanes_difference(gamma1_value, gamma2_value, baseline_lane, time_col_index, value_col_index, timestep_interval)
        else:
            # Lanes comparison mode (all available lanes for given gamma1, gamma2)
            if len(sys.argv) < 3:
                print("Error: Please provide gamma1 and gamma2 values. Usage: python script.py gamma1_value gamma2_value [time_col] [value_col] [timestep_interval]")
                return
            
            gamma1_value = sys.argv[1]
            gamma2_value = sys.argv[2]
            time_col_index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
            value_col_index = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1
            timestep_interval = int(float(sys.argv[5])) if len(sys.argv) > 5 else 1000
            plot_lanes_comparison(gamma1_value, gamma2_value, time_col_index, value_col_index, timestep_interval)
    else:
        # Interactive mode
        print("Choose mode:")
        print("1. Plot comparison for all lane values for a gamma1 and gamma2")
        print("2. Plot comparison for specific lane values for a gamma1 and gamma2")
        print("3. Plot comparison for a single network type across specific lane values")
        print("4. Plot comparison for a single network type across ALL lane values")
        print("5. Plot difference from baseline lane (e.g., Lanes 3-2, Lanes 4-2)")
        
        choice = input("Enter choice (1, 2, 3, 4, or 5): ").strip()
        
        if choice == "1":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0375): ").strip()
            gamma2_value = input("Enter gamma2 value (e.g., 0.1000): ").strip()
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_lanes_comparison(gamma1_value, gamma2_value, time_col_index, value_col_index, timestep_interval)
        elif choice == "2":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0375): ").strip()
            gamma2_value = input("Enter gamma2 value (e.g., 0.1000): ").strip()
            lane_str = input("Enter lane values separated by commas (e.g., 2, 3, 4): ").strip()
            lane_values = [l.strip() for l in lane_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_specific_lanes_comparison(gamma1_value, gamma2_value, lane_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "3":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0375): ").strip()
            gamma2_value = input("Enter gamma2 value (e.g., 0.1000): ").strip()
            network_type = input("Enter network type (e.g., 'PM 5', 'AM Base', 'AM 2', etc.): ").strip()
            lane_str = input("Enter lane values separated by commas (e.g., 2, 3, 4): ").strip()
            lane_values = [l.strip() for l in lane_str.split(',')]
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_single_network_lanes_comparison(gamma1_value, gamma2_value, network_type, lane_values, time_col_index, value_col_index, timestep_interval)
        elif choice == "4":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0375): ").strip()
            gamma2_value = input("Enter gamma2 value (e.g., 0.1000): ").strip()
            network_type = input("Enter network type (e.g., 'PM 5', 'AM Base', 'AM 2', etc.): ").strip()
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_all_lanes_single_network(gamma1_value, gamma2_value, network_type, time_col_index, value_col_index, timestep_interval)
        elif choice == "5":
            gamma1_value = input("Enter gamma1 value (e.g., 0.0375): ").strip()
            gamma2_value = input("Enter gamma2 value (e.g., 0.1000): ").strip()
            baseline_str = input("Enter baseline lane number (default 2): ").strip()
            baseline_lane = int(baseline_str) if baseline_str else 2
            
            time_col_str = input("Enter time column index (default 0): ").strip()
            value_col_str = input("Enter value column index (default 1 for weighted_integrated_cars): ").strip()
            timestep_str = input("Enter timestep interval (default 1000): ").strip()
            
            time_col_index = int(time_col_str) if time_col_str else 0
            value_col_index = int(value_col_str) if value_col_str else 1
            timestep_interval = int(float(timestep_str)) if timestep_str else 1000
            
            plot_lanes_difference(gamma1_value, gamma2_value, baseline_lane, time_col_index, value_col_index, timestep_interval)
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
