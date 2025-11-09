from collections import defaultdict
from road import Road
from junction import Junction
import heapq
import numpy as np

class Network(object):
    def __init__(self, junctions: list[Junction]):
        self.junctions = junctions
        self.check_road_names_different()
        self.roads : list[Road] = []

        for junction in junctions:
            for r in junction.roads_in + junction.roads_out:
                if r not in self.roads:
                    self.roads.append(r)

    def __eq__(self, other):
        if isinstance(other, Network):
            return self.roads == other.roads
        return False

    def update_roads(self):
        """
        Updates the roads field for a network object by searching through the roads in its junctions. This is useful when the roads have been modified and we want to update the network accordingly.
        """
        self.roads = []
        for junction in self.junctions:
            for r in junction.roads_in + junction.roads_out:
                if r not in self.roads:
                    self.roads.append(r)

    def get_junctions(self):
        return self.junctions
    
    def get_roads(self):
        return self.roads

    def reset_by_func(self):
        for junction in self.junctions:
            junction.reset_by_func()

    def reset_by_data(self):
        for junction in self.junctions:
            junction.reset_by_data()
    
    def resolve(self):
        """
        Calls resolve on all junctions and boundary conditions in the network.
        """
        for junction in self.junctions:
            junction.boundary_resolve()
        for junction in self.junctions:
            junction.resolve()

    def evolve_resolve(self, record_densities = False):
        """
        Calls evolve on all junctions in the network, then calls resolve on all junctions in the network.
        """
        for junction in self.junctions:
            junction.evolve()
        for junction in self.junctions:
            junction.boundary_resolve()
        for junction in self.junctions:
            junction.resolve()

        if record_densities:
            for r in self.roads:
                r.record_road()

    def modify_urgency(self, roads, new_urgency):
        for r in self.roads:
            if r in roads:
                r_sigma = r.get_sigma()
                r.set_left_boundary_function(lambda time,sig = r_sigma, urg = new_urgency: urg)
                r.set_initial_density_func(lambda x, sig = r_sigma, urg = new_urgency: urg*np.ones_like(x))

    def remove_road(self, r: Road):
        """
        Removes road r from all junctions in the network. It is okay if r is already removed from the network.
        """
        self.roads.remove(r)
        for junction in self.junctions:
            if r in junction.roads_in:
                junction.roads_in.remove(r)
            if r in junction.roads_out:
                junction.roads_out.remove(r)

    def read_preferences(self, preferences_txt):
        ''' Reads preferences from preferences_txt. If preferences_txt is None, uses default parameters. '''
        if preferences_txt is None:
            self.set_preferences(None)
        else:
            with open(preferences_txt, 'r') as f:
                lines = [float(line.strip()) for line in f]
            self.set_preferences(np.array(lines))

    def set_preferences(self, prefs):
        ''' Sets preferences in the order that junctions were provided. If prefs is None, uses default parameters. '''
        if prefs is None:
            prefs = self.get_default_preferences()

        index = 0
        for junction in self.junctions:
            junction.set_preferences(prefs[index:index + junction.get_dim()])
            index += junction.get_dim()

    def get_default_preferences(self):
        default_preferences = []
        for junction in self.junctions:
            default_preferences.extend(junction.get_dim()*[1/len(junction.get_roads_out()),])
        return default_preferences

    def largest_preference(self):
        return max(junction.largest_preference() for junction in self.junctions if junction.get_dim() > 0)
    
    def smallest_preference(self):
        return min(junction.smallest_preference() for junction in self.junctions if junction.get_dim() > 0)
    
    def get_time_integrated_cars_exited(self):
        return sum(r.time_integrated_cars_exited for r in self.roads if r.is_exit_road)
    
    def get_time_integrated_cars_entered(self):
        return sum(r.time_integrated_cars_exited for r in self.roads if r.left_junction is None)
      
    def get_time_integrated_cars_on_exit_roads(self):
        return sum(r.time_integrated_cars_on_the_road for r in self.roads if r.is_exit_road)
    
    def get_time_integrated_cars_on_main_roads(self):
        """
        Gets the time integrated cars on the main roads, where a main road is any road in the set 
        {Roads} - {sources} - {exits}.
        """
        return sum(r.time_integrated_cars_on_the_road for r in self.roads if not r.is_exit_road and r.left_junction is not None)
    
    def compute_distances(self, j:Junction):
        """
        Use Dijkstra's to compute the distance from each junction to the j junction.
        """
        # Priority queue to store the junctions to be visited
        pq = [(1, id(j), j)]
        # Record distances to all junctions
        self.distances = {node: float('inf') for node in self.junctions}
        # Initializing distance to the source junction j to be zero
        self.distances[j] = 1 # We assume that by default, we run this for j = h5, which is the closest to the exiting junction.
        
        while pq:
            current_distance, _, current_node = heapq.heappop(pq)
            
            if current_distance > self.distances[current_node]:
                continue
            
            for r in current_node.get_roads_in():
                junc = r.left_junction
                if junc == None:
                    continue
                distance = current_distance + 1
                if distance < self.distances[junc]:
                    self.distances[junc] = distance
                    heapq.heappush(pq, (distance, id(junc), junc))
        
        return self.distances
    
    def compute_distances_mult_exits(self, j_list: list[Junction]):
        self.distances_1 = self.compute_distances(j_list[0])
        num_exits = len(j_list)
        for i in range(1,num_exits):
            self.distances_2 = self.compute_distances(j_list[i])
            for j in self.distances_1:
                if self.distances_1[j] > self.distances_2[j]:
                    self.distances[j] = self.distances_2[j]
                else:
                    self.distances[j] = self.distances_1[j]
            self.distances_1 = self.distances
        return self.distances

    
    def distance_map(self, d):
        return (0.5)**(d)
    
    def get_time_integrated_cars_distance_scaled(self):
        # To count every single road only once, we count all roads going into each junction + the exiting roads (only)
        return sum(r.time_integrated_cars_on_the_road*self.distance_map(self.distances[r.right_junction]) for r in self.roads if not r.is_exit_road and r.right_junction is not None) \
            + sum(r.time_integrated_cars_on_the_road for r in self.roads if r.is_exit_road)

    def all_reached_LOS(self, LOS):
        '''
            Returns true if all roads at every point on them are at least at the inputted LOS.
            Skips checking exit roads.
        '''
        match LOS:
            case "A": i = 0
            case "B": i = 1
            case "C": i = 2
            case "D": i = 3
            case "E": i = 4
            case "F": i = 5
            case _: print("Error in LOS checking, only inputs A - F accepted")
        
        for r in self.roads:
            if r.is_exit_road: # skip checking the exit road
                continue
            num = 0 if i == 0 else r.loc[i-1]
            if not np.all(r.current_density[1:-1] >= num):
                return False

        return True
    
    def partially_reached_LOS(self, LOS):
        '''
            Returns true if all roads at at least one point on them are at least at the inputted LOS.
            Skips checking exit roads.
        '''
        match LOS:
            case "A": i = 0
            case "B": i = 1
            case "C": i = 2
            case "D": i = 3
            case "E": i = 4
            case "F": i = 5
            case _: print("Error in LOS checking, only inputs A - F accepted")
        
        for r in self.roads:
            if r.is_exit_road: # skip checking the exit road
                continue
            num = 0 if i == 0 else r.loc[i-1]
            if not np.any(r.current_density[1:-1] >= num):
                return False

        return True
    
    def half_reached_LOS(self, LOS):
        '''
            Returns true if all roads at half the points on them are at least at the inputted LOS.
            Skips checking exit roads.
        '''
        match LOS:
            case "A": i = 0
            case "B": i = 1
            case "C": i = 2
            case "D": i = 3
            case "E": i = 4
            case "F": i = 5
            case _: print("Error in LOS checking, only inputs A - F accepted")
        
        for r in self.roads:
            if r.is_exit_road: # skip checking the exit road
                continue
            num = 0 if i == 0 else r.loc[i-1]
            mask = r.current_density[1:-1] >= num
            fraction = np.mean(mask)
            
            if fraction < 0.5:
                return False

        return True
    
    def count_roads_per_LOS(self):
        '''
            Returns a dictionary with the number of roads at each LOS level.
            Each road's LOS is determined by its average density across all road segments
            that make up that road (e.g., hwy30 has multiple segments, the average is computed
            across all non-exit segments in hwy30).
            Skips individual exit road segments but includes the road if it has other segments.
        '''
        # Initialize the dictionary with all LOS levels
        los_levels = ["A", "B", "C", "D", "E", "F"]
        los_count = {level: 0 for level in los_levels}
        
        # Group road segments by their parent road
        road_groups = self.group_road_segments_by_road()
        
        for road_name, road_segments in road_groups.items():
            # Calculate the average LOS across all non-exit segments in this road
            total_los = 0
            valid_segments = 0
            
            for segment in road_segments:
                if not segment.is_exit_road:  # Skip only the exit road segments
                    segment_los = segment.compute_average_los()
                    total_los += segment_los
                    valid_segments += 1
            
            if valid_segments > 0:
                # Compute average LOS for the entire road (excluding exit segments)
                avg_road_los = total_los / valid_segments
                
                # Determine which LOS category this falls into
                # avg_road_los is a float where 0-1 = A, 1-2 = B, etc.
                los_index = int(avg_road_los)
                los_index = min(los_index, 5)  # Cap at F (index 5)
                los_index = max(los_index, 0)  # Floor at A (index 0)
                
                los_count[los_levels[los_index]] += 1
            
        return los_count
    
    def count_segments_per_LOS(self):
        '''
            Returns a dictionary with the number of road segments at each LOS level.
            Each road segment's LOS is determined by its average density 
            Skips individual exit road segments but includes the road if it has other segments.
        '''
        # Initialize the dictionary with all LOS levels
        los_levels = ["A", "B", "C", "D", "E", "F"]
        los_count = {level: 0 for level in los_levels}
        
        for segment in self.roads:
            # Calculate the average LOS across all non-exit segments in network
            segment_los = 0
            if not segment.is_exit_road:
                segment_los = segment.compute_average_los()
                # Determine which LOS category this falls into
                # segment_los should be an int corresponding to LOS directly, but just in case
                los_index = int(segment_los)
                los_index = min(los_index, 5)  # Cap at F (index 5)
                los_index = max(los_index, 0)  # Floor at A (index 0)
                    
                los_count[los_levels[los_index]] += 1
            
        return los_count
    
    def group_road_segments_by_road(self):
        '''
            Returns a dictionary that groups road segments by the road they belong to.
            The key is the road name (extracted from the road segment name), and the value
            is a list of road segments that belong to that road.
            
            Road segment names are now in the format: "road_name index: description"
            e.g., "hwy30 0: prison -> dickenson", "lahainaluna left 5: kale -> paunau"
        '''
        road_groups = defaultdict(list)
        
        for road_segment in self.roads:
            segment_name = road_segment.name
            
            # road names are of the form hwy30 0: prison -> dickenson
            if ":" in segment_name:
                # Split on the colon to separate the road identifier from the description
                road_identifier = segment_name.split(":")[0].strip()
                
                # Extract the road name by removing the index number at the end
                parts = road_identifier.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    # Remove the last part (index) to get the road name
                    road_name = " ".join(parts[:-1])
                else:
                    # If no index found, use the entire identifier
                    road_name = road_identifier
            else:
                # Fallback for roads that don't follow the new format
                # For individual roads (sources/exits), use the road name itself
                road_name = segment_name
            
            road_groups[road_name].append(road_segment)
        
        # Convert defaultdict to regular dict and sort segments within each road by their index
        result = {}
        for road_name, segments in road_groups.items():
            # Sort segments by the index number in their name
            def extract_segment_index(road_segment):
                try:
                    segment_name = road_segment.name
                    if ":" in segment_name:
                        # Extract index from "road_name index: description" format
                        road_identifier = segment_name.split(":")[0].strip()
                        parts = road_identifier.split()
                        if len(parts) >= 2 and parts[-1].isdigit():
                            return int(parts[-1])
                    return 0
                except:
                    return 0
            
            sorted_segments = sorted(segments, key=extract_segment_index)
            result[road_name] = sorted_segments
        
        return result

    def get_dim(self):
        """
        Returns the dimension of the preference parameters across the network, NOT the total number of preferences.
        E.x. for a network consisting of one 2in-4out junction, returns 2*(4-1)  = 6, not 8.
        """
        return sum(junction.get_dim() for junction in self.junctions)
    
    def get_preference_info(self, index):
        """
        Given an index in the flattened preference vector, returns information about
        which junction this preference belongs to and which incoming/outgoing road pair it represents.
        
        Arguments:
        - index: The index in the flattened preference vector (0-based)
        
        Returns:
        - dict: {
            'junction': Junction object,
            'junction_name': str,
            'incoming_road': Road object,
            'outgoing_road': Road object,
            'incoming_road_index': int (index within junction's roads_in),
            'outgoing_road_index': int (index within junction's roads_out),
            'local_preference_index': int (index within this junction's preferences),
            'preference_description': str (human-readable description)
        }
        
        Raises:
        - IndexError: If index is out of bounds for the preference vector
        """
        total_dim = self.get_dim()
        if index < 0 or index >= total_dim:
            raise IndexError(f"Index {index} is out of bounds for preference vector of dimension {total_dim}")
        
        current_index = 0
        
        for junction in self.junctions:
            junction_dim = junction.get_dim()
            
            # Check if the index falls within this junction's preferences
            if current_index <= index < current_index + junction_dim:
                # Calculate local index within this junction
                local_index = index - current_index
                
                # Calculate which incoming road and outgoing road this preference corresponds to
                num_out = len(junction.roads_out)
                num_in = len(junction.roads_in)
                prefs_per_incoming = num_out - 1  # We store (num_out - 1) preferences per incoming road
                
                # Find which incoming road this preference belongs to
                incoming_road_idx = local_index // prefs_per_incoming
                # Find which outgoing road this preference points to
                outgoing_road_idx = local_index % prefs_per_incoming
                
                incoming_road = junction.roads_in[incoming_road_idx]
                outgoing_road = junction.roads_out[outgoing_road_idx]
                
                # Create description
                description = f"Traffic from '{incoming_road.name}' to '{outgoing_road.name}' at junction '{junction.name}'"
                
                return {
                    'junction': junction,
                    'junction_name': junction.name,
                    'incoming_road': incoming_road,
                    'outgoing_road': outgoing_road,
                    'incoming_road_index': incoming_road_idx,
                    'outgoing_road_index': outgoing_road_idx,
                    'local_preference_index': local_index,
                    'preference_description': description
                }
            
            current_index += junction_dim
        
        # This should never be reached if index bounds checking is correct
        raise IndexError(f"Index {index} could not be mapped to any junction preference")
    
    def get_preferences_summary(self):
        """
        Returns a comprehensive summary of all preferences in the network.
        
        Returns:
        - list: List of dictionaries, each containing information about one preference parameter.
                Each dict has the same structure as returned by get_preference_info().
        """
        summary = []
        total_dim = self.get_dim()
        
        for i in range(total_dim):
            summary.append(self.get_preference_info(i))
        
        return summary
    
    def print_preferences_summary(self):
        """
        Prints a human-readable summary of all preferences in the network.
        """
        summary = self.get_preferences_summary()
        
        print(f"Network Preference Vector Summary (Total dimension: {self.get_dim()})")
        print("=" * 80)
        
        current_junction = None
        for i, pref_info in enumerate(summary):
            # Print junction header when we encounter a new junction
            if current_junction != pref_info['junction_name']:
                current_junction = pref_info['junction_name']
                print(f"\nJunction: {current_junction}")
                print("-" * 40)
            
            print(f"  Index {i:3d}: {pref_info['preference_description']}")
        
        print("=" * 80)
    
    def check_num_junctions(self, num):
        """ 
        Quick sanity check that the number of junctions in the network is correct. Used to make sure that all the expected junctions have been inputted into the network.
        """
        if len(self.junctions) != num:
            raise Exception("Found " + str(len(self.junctions)) + " junctions! Expected " + str(num) + ".")
        #print("number of junctions check successful!")

    def check_roads_in_roadlist(self, road_list: list["Road"]):
        """
        Checks three conditions.
        1. Checks that every road used in a junction in the network is in road_list. Used to make sure that the inputted road_list contains all expected roads.
        2. Checks that every road in road_list is utilized completely. That is, every road in road_list is incoming at some junction (or has a right boundary function)
        and outgoing at some junction (or has a left boundary function). Used to make sure that a road in road_list wasn't forgotten to be included at a junction.
        3. Checks that every junction of a road in road_list is in the network. Used to make sure that the network contains all expected junctions.

        Raises an Exception if the check fails.
        """

        # checking condition 1
        for junction in self.junctions:
            for r in junction.roads_in + junction.roads_out:
                if r not in road_list:
                    raise Exception(str(r) + " is not in road list.")

        # checking condition 2 - Some roads might not have the left/right boundary function defined, 
        # so we shouldn't check for those (since we have an automatic "non-reflecting boundary" condition appllied)
        # i modified roads so that reflecting conditions have to be set by setting the boundary function to be the string "reflecting"
        for r in road_list:
            if r.left_junction is None and r.left_boundary_function is None:
                raise Exception(str(r) + " left junction not well defined")
            if r.right_junction is None and r.right_boundary_function is None:
                raise Exception(str(r) + " right junction not well defined")
            
        # checking condition 3
        for r in road_list:
            if r.left_junction is not None and r.left_junction not in self.junctions:
                raise Exception(str(r) + " left junction not found in network's junctions")
            if r.right_junction is not None and r.right_junction not in self.junctions:
                raise Exception(str(r) + " right junction not found in network's junctions")

        #print("network check successful!")

    def check_road_names_different(self):
        first = set()
        second = set()
        for junction in self.junctions:
            for road in junction.roads_in:
                if road.name in first:
                    if road.name in second:
                        raise Exception("duplicate name " + road.name)
                    second.add(road.name)
                first.add(road.name)

    @staticmethod
    def prepare_network_densities(network_from: 'Network', network_to: 'Network'):
        '''Prepares the initial densities in network_to, using the data from network_from.'''
        # Copy in data from network_from into network_to
        # Optimized approach using a dictionary to map names to roads in road_list_to to road_list_from
        # These roads are obtained by reference
        
        if network_from is None:
           network_to.reset_by_data()
        else:
            AM_SPECIAL_MAP_NAMES = {"hwy30 0: source -> prison": "hwy30 source -> lahainaluna",
                                    "hwy30 1: prison -> dickenson": "hwy30 source -> lahainaluna",
                                    "hwy30 2: dickenson -> lahainaluna": "hwy30 source -> lahainaluna",
                                    "front 0: source -> prison": "front source -> lahainaluna",
                                    "front 1: prison -> canal": "front source -> lahainaluna",
                                    "front 2: canal -> dickenson": "front source -> lahainaluna",
                                    "front 3: dickenson -> lahainaluna": "front source -> lahainaluna",
                                    "wainee 0: source -> prison": "wainee source -> lahainaluna",
                                    "wainee 1: prison -> hale": "wainee source -> lahainaluna",
                                    "wainee 2: hale -> dickenson": "wainee source -> lahainaluna",
                                    "wainee 3: dickenson -> panaewa": "wainee source -> lahainaluna",
                                    "wainee 4: panaewa -> lahainaluna": "wainee source -> lahainaluna"}
            road_map_index = defaultdict(int)
            road_list_from = network_from.get_roads()
            road_list_to = network_to.get_roads()
            
            for r in road_list_to:
                for r2 in road_list_from:
                    if r.name in AM_SPECIAL_MAP_NAMES:
                        if r2.name == AM_SPECIAL_MAP_NAMES[r.name]:
                            arr = r2.get_current_density()[road_map_index[r2.name]:road_map_index[r2.name]+len(r.cells)]
                            r.set_initial_density(arr)
                            road_map_index[r2.name] += len(r.cells) - 2
                            break
                    if r == r2:
                        r.set_initial_density(r2.get_current_density())
                        break

    @staticmethod
    def prepare_network(network_from: 'Network', network_to: 'Network', preferences_txt = None):
        Network.prepare_network_densities(network_from, network_to)
        network_to.read_preferences(preferences_txt)
