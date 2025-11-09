"""
Simple script to run a network and create an animation of the traffic density over time.
"""
from matplotlib.animation import FFMpegWriter
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import os

from network import Network
from roads_setup import *
from am_networks_setup import *
from pm_networks_setup import *

colors = [
    "#035303",  # 0 - dark green
    "#7fff00",  # 1 - lighter green
    "#ffff00",  # 2 - yellow
    "#ff9900",  # 3 - orange-red
    "#ff3300",  # 4 - red-orange
    "#cc0000",  # 5 - deep red
]

# Create a colormap from the list
cmp = LinearSegmentedColormap.from_list("cmp", colors, N=30)
JUNCTION_COLOR = [0, 0, 0]

# Load images
image = Image.open("Lahaina_network_info/blank_map.png").convert("RGB")
junctions = Image.open("Lahaina_network_info/junctions.png").convert("RGB")
left_right = Image.open("Lahaina_network_info/left_right.png").convert("RGB")
up_down = Image.open("Lahaina_network_info/up_down.png").convert("RGB")
label_map = np.load("Lahaina_network_info/label_map.npy")

height, width = label_map.shape

# Convert image to numpy array
image_array = np.array(image)

all_roads = create_all_roads(0)

# Unpack main groups so the roads are accessible like before
hwy30, luna_left, luna_right, front, wainee, keawe, bypass, \
kenui_left, kenui_right, papalaua_left, papalaua_right, \
dickenson_left, dickenson_right, prison_left, prison_right, \
kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad, \
gateway, oilroad, puanoa_pl, baker_left, wahie, canal, \
baker_right, panaewa, hale, kahoma_village, luakini_up, luakini_down = all_roads

hwy30_source, front_source, wainee_source = create_am_sources(0)
hwy30_down, bypass_down, front_down, wainee_down = create_pm_special_roads(hwy30, bypass, front, wainee)

ROAD_LIST = (hwy30
              + luna_left[0:-1] + luna_right
              + front + wainee
              + bypass + keawe
              + kenui_left[0:-1] + kenui_right[0:-1]
              + papalaua_left + papalaua_right
              + [puanoa_pl, kahoma_village, baker_left, baker_right, wahie]
              + [komomai, kuhua, pauoa, kale, paunau_st, kalena]
              + [luakini_down, luakini_up, hale, canal, panaewa]
              + dickenson_left + dickenson_right
              + prison_left + prison_right
              + hwy30_down + [bypass_down] + front_down + wainee_down
            )

# read in labels
with open("Lahaina_network_info/labels_to_road_names.txt", "r") as f:
    label_to_road = [next(r for r in ROAD_LIST if r.name == line.strip()) for line in f]

FLIP = [r for r in ROAD_LIST if r not in label_to_road]

# read in pixel ratios
with open("Lahaina_network_info/pixel_ratios.json", "r") as f:
    loaded_json = json.load(f)
    # Convert keys back to tuples
    pixel_to_ratio = {tuple(map(int, k.split(","))): v for k, v in loaded_json.items()}

with open("Lahaina_network_info/special_pixel_ratios.json", "r") as f:
    loaded_json = json.load(f)
    # Convert keys back to tuples
    special_pixel_to_ratio = {tuple(map(int, k.split(","))): v for k, v in loaded_json.items()}

LEFT = luna_left[0:-1] + papalaua_left + kenui_left[0:-1] + dickenson_left + prison_left
RIGHT = luna_right[0:-1] + papalaua_right + kenui_right[0:-1] + dickenson_right + prison_right
UP = hwy30[0:6] + [bypass[0]] + front[0:10] + wainee[0:8] 
DOWN = hwy30_down + [bypass_down] + front_down + wainee_down

ROAD_JUNCT = Road('JUNCTION_')

AM_NETWORKS = setup_am_networks(0)
KEAWE_BYPASS_EXCLUDED_NETWORKS = setup_pm_networks(0)[1:]

AM_SPECIAL = (hwy30_source, front_source, wainee_source)

AM_SPECIAL_MAP = {
    hwy30[0]: hwy30_source,
    hwy30[1]: hwy30_source,
    hwy30[2]: hwy30_source,
    front[0]: front_source,
    front[1]: front_source,
    front[2]: front_source,
    front[3]: front_source,
    wainee[0]: wainee_source,
    wainee[1]: wainee_source,
    wainee[2]: wainee_source,
    wainee[3]: wainee_source,
    wainee[4]: wainee_source
}

# Set up map from pixel to road

def setup_pixel_map(network):
    pixel_map = {}

    for x in range(width):
        for y in range(height):
            if junctions.getpixel((x, y)) == (255, 0, 0):
                # junction
                if network not in AM_NETWORKS or y < 567 or (y < 610 and x < 350):
                    if network in KEAWE_BYPASS_EXCLUDED_NETWORKS and (x - 462)**2 + (y - 180)**2 < 100:
                        continue
                    pixel_map[(x, y)] = ROAD_JUNCT
                    continue
            
            l = label_map[y, x]
            if l > 0:
                road = label_to_road[l - 1]
                
                if road in AM_SPECIAL_MAP:
                    if network in AM_NETWORKS:
                        road = AM_SPECIAL_MAP[road]

                if road in RIGHT:
                    # determine if in left or right by doing a search for blue pixel being above or below
                    for i in range(10):
                        c = left_right.getpixel((x, y - i))
                        if c[1] < 238 and c[2] > 144: # blue detected below
                            if LEFT[RIGHT.index(road)] in network.roads:
                                road = LEFT[RIGHT.index(road)]
                            break

                if road in UP:
                    for i in range(10):
                        c = up_down.getpixel((x + i, y))
                        if c[1] < 238 and c[2] > 144:
                            if DOWN[UP.index(road)] in network.roads:
                                road = DOWN[UP.index(road)]
                            break
                
                for r in network.roads:
                    if road == r:
                        pixel_map[(x, y)] = r
                            
    return pixel_map

def get_density(x, y, road):
    if road in AM_SPECIAL:
        ratio = special_pixel_to_ratio[(y, x)]
    else:
        ratio = pixel_to_ratio[(y, x)]

    if road in FLIP:
        ratio = 1 - ratio

    index = math.floor(ratio * (len(road.current_density) - 3)) + 1 # index into density array
    return road.compute_los_level(road.current_density[index]) / 5

# Function for getting image
def get_image_array(pixel_map):
    heatmap_image = np.copy(image_array)

    for (x,y), road in pixel_map.items():
        if road == ROAD_JUNCT:
              heatmap_image[y][x] = JUNCTION_COLOR
        else:
            heatmap_image[y][x] = cmp([get_density(x, y, road)])[0,:3]*255

    return heatmap_image

def run_network(network_from, network_to: Network, nt, preferences_txt=None, output_dir=None, create_animation = False, step = 600, create_picture = False, tag = ""):
    '''
    Runs network_to for nt steps.

    Initializes network_to based on the data in network_from.

    Reads the preferences from preferences_txt. If no preferences_txt is specified, then the default parameters are used.

    If output_dir is specified, saves time series data to this directory.

    Creates an animation with specified step rate if create_animation is True. Saves to output_dir if specified.

    Creates an image of the final state if create_picture is True. Saves to output_dir if specified.
    '''
    Network.prepare_network(network_from, network_to, preferences_txt)
    # double check the network road grouping is working
    # road_groups = network_to.group_road_segments_by_road()

    # Setup CSV output if output directory is provided
    csv_file_path = None
    dt = 0.1  # Road.dt value (0.1 seconds per step)
    
    if output_dir:
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        csv_file_path = os.path.join(output_dir, "time_series_data" + tag + ".csv")
        with open(csv_file_path, "w") as f:
            f.write("time_seconds, weighted_integrated_cars, cars_entered, cars_exited, time_int_cars_exits, time_int_cars_main, LOSA_count, LOSB_count, LOSC_count, LOSD_count, LOSE_count, segment_LOSA_count, segment_LOSB_count, segment_LOSC_count, segment_LOSD_count, segment_LOSE_count\n")
        print(f"Saving time series data to: {csv_file_path}")
    
    print("Starting simulation...")
    if create_animation or create_picture:
        print("Setting up pixel map...")
        pixel_map = setup_pixel_map(network_to) # Setup pixel map

    if create_animation:
        print("Preparing animation...")
        animation_file_path = os.path.join(output_dir, "animation" + tag + ".mp4")

        # Prepare the figure and axis
        fig, ax = plt.subplots()
        heatmap_image = get_image_array(pixel_map)
        im = ax.imshow(heatmap_image)
        
        writer = FFMpegWriter(fps=20, metadata=dict(artist='me'))
        
        with writer.saving(fig, animation_file_path, dpi=150):
            for i in range(nt):
                if i % step == 0:
                    print("Evolving network for frame " + str(i) + " of " + str(nt))
                
                if i % step == 0:
                    heatmap_image = get_image_array(pixel_map)
                    im.set_array(heatmap_image)  # Update image data
                    ax.set_title(f"nt = {i}")
                    writer.grab_frame()
                network_to.evolve_resolve()
                
                # Save data every "step" iterations (if output directory specified)
                if csv_file_path is not None and i % step == 0:
                    time_seconds = i * dt  # Convert to actual seconds
                    weighted_cars = network_to.get_time_integrated_cars_distance_scaled()
                    los_count = network_to.count_roads_per_LOS()
                    segment_los_count = network_to.count_segments_per_LOS()
                    cars_entered = network_to.get_time_integrated_cars_entered()
                    cars_exited = network_to.get_time_integrated_cars_exited()
                    time_int_cars_exits = network_to.get_time_integrated_cars_on_exit_roads()
                    time_int_cars_main = network_to.get_time_integrated_cars_on_main_roads()
                    with open(csv_file_path, "a") as f:
                        f.write(f"{time_seconds}, {weighted_cars}, {cars_entered}, {cars_exited}, {time_int_cars_exits}, {time_int_cars_main}, {los_count['A']}, {los_count['B']}, {los_count['C']}, {los_count['D']}, {los_count['E']}, {segment_los_count['A']}, {segment_los_count['B']}, {segment_los_count['C']}, {segment_los_count['D']}, {segment_los_count['E']}\n")
                if i % 3000 == 0:
                    heatmap_image = get_image_array(pixel_map)
                    heatmap_image = Image.fromarray(heatmap_image)
                    image_path = f"step_{i}" + "picture" + tag + ".png"
                    heatmap_image.save(os.path.join(output_dir, image_path))

    else:                   
        for i in range(nt):
            if i % step == 0:
                print("Evolving network for frame " + str(i) + " of " + str(nt))
            
            network_to.evolve_resolve()
            
            # Save data every "step" iterations (if output directory specified)
            if csv_file_path is not None and i % step == 0:
                    time_seconds = i * dt  # Convert to actual seconds
                    weighted_cars = network_to.get_time_integrated_cars_distance_scaled()
                    los_count = network_to.count_roads_per_LOS()
                    segment_los_count = network_to.count_segments_per_LOS()
                    cars_entered = network_to.get_time_integrated_cars_entered()
                    cars_exited = network_to.get_time_integrated_cars_exited()
                    time_int_cars_exits = network_to.get_time_integrated_cars_on_exit_roads()
                    time_int_cars_main = network_to.get_time_integrated_cars_on_main_roads()
                    with open(csv_file_path, "a") as f:
                        f.write(f"{time_seconds}, {weighted_cars}, {cars_entered}, {cars_exited}, {time_int_cars_exits}, {time_int_cars_main}, {los_count['A']}, {los_count['B']}, {los_count['C']}, {los_count['D']}, {los_count['E']}, {segment_los_count['A']}, {segment_los_count['B']}, {segment_los_count['C']}, {segment_los_count['D']}, {segment_los_count['E']}\n")
    # Save final data point

    if create_picture:
        heatmap_image = get_image_array(pixel_map)
        heatmap_image = Image.fromarray(heatmap_image)
        heatmap_image.save(os.path.join(output_dir, "picture" + tag + ".png"))

    if csv_file_path is not None:
        final_time = nt * dt
        final_cars = network_to.get_time_integrated_cars_distance_scaled()
        final_los_count = network_to.count_roads_per_LOS()
        final_segment_los_count = network_to.count_segments_per_LOS()
        final_cars_entered = network_to.get_time_integrated_cars_entered()
        final_cars_exited = network_to.get_time_integrated_cars_exited()
        final_time_int_cars_exits = network_to.get_time_integrated_cars_on_exit_roads()
        final_time_int_cars_main = network_to.get_time_integrated_cars_on_main_roads()
        with open(csv_file_path, "a") as f:
                f.write(f"{final_time}, {final_cars}, {final_cars_entered}, {final_cars_exited}, {final_time_int_cars_exits}, {final_time_int_cars_main}, {final_los_count['A']}, {final_los_count['B']}, {final_los_count['C']}, {final_los_count['D']}, {final_los_count['E']}, {final_segment_los_count['A']}, {final_segment_los_count['B']}, {final_segment_los_count['C']}, {final_segment_los_count['D']}, {final_segment_los_count['E']}\n")
        print(f"Final weighted integrated cars: {final_cars:.6f}")
        print(f"Time series data saved to: {csv_file_path}")

# Run experiment (only if this script is run directly, not imported)
if __name__ == "__main__":
    networks = setup_am_networks(0.05)  # Reset networks to ensure clean state
    network_am_base = networks[0]
    run_network(None, network_am_base, nt=1000, output_dir="base_script", create_animation=True, step=500, create_picture=True, tag="_AM_Base")
