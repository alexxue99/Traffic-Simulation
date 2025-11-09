import numpy as np
from road import Road

class Junction(object):
    def __init__(self,name):
        self.name = name
        self.roads_in = None
        self.roads_out = None
        self.is_exit_junction = False
        self.preferences = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        if isinstance(other, Junction):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def get_name(self):
        return self.name

    def reset_by_func(self):
        for r in self.roads_in + self.roads_out:
            r.reset_by_func()

    def reset_by_data(self):
        for r in self.roads_in + self.roads_out:
            r.reset_by_data()

    def evolve(self):
        for r in self.roads_in:
            r.evolve()
        # This is almost sufficient since every road is an incoming road for some junction.
        # However, we need to evolve the exit roads separately (since it is not an incoming road for any junction).
        for r in self.roads_out:
            if r.is_exit_road:
                r.evolve()
    
    def boundary_resolve(self):
        for r in self.roads_in + self.roads_out:
            r.boundary_resolve()
    
    def record(self):
        for r in self.roads_in + self.roads_out:
            r.record_road()

    def set_is_exit_junction(self):
        self.is_exit_junction = True
        for road in self.roads_out:
            road.set_is_exit_road()

    def set_roads_in(self, *roads: Road):
        self.roads_in = list(roads)

        for road in roads:
            if road.right_junction is not None:
                raise Exception(str(road) + " already has right junction " + str(road.right_junction))
            if road.right_boundary_function is not None:
                raise Exception(str(road) + " already has right boundary function")
            road.right_junction = self
        return self 
    
    def set_roads_out(self, *roads: Road):
        self.roads_out = list(roads)

        for road in roads:
            if road.left_junction is not None:
                raise Exception(str(road) + " already has left junction " + str(road.left_junction))
            if road.left_boundary_function is not None:
                raise Exception(str(road) + " already has left boundary function")
            road.left_junction = self
            
        return self 
    
    def get_dim(self):
        return (len(self.roads_out) - 1) * len(self.roads_in)
    
    def get_num_in(self):
        return len(self.roads_in)
    
    def get_num_out(self):
        return len(self.roads_out)
    
    def set_preferences(self, *preferences):
        """
        Note that the input "preferences" would be a list, while the output "preferences" (under self.preferences) would be a 2d-array.
        """
        if len(self.roads_out) == 1:
            self.preferences = [[1] for _ in range(len(self.roads_in))]
            return self
        """
        Preferences are listed in order of: all prefs (excluding last pref) for incoming road 0,
        then all prefs (excluding last pref) for incoming road 1, e.t.c.
        """
        if isinstance(preferences[0], list):
            l = preferences[0]
        elif isinstance(preferences[0], np.ndarray):
            l = preferences[0].tolist()
        else:
            l = list(preferences)
        self.preferences = [] # will be a 2d-array where the (i,j)th element indicates preference for ith incoming road -> jth outgoing road

        index = 0
        length = len(self.roads_out) - 1
        while index < len(l):
            in_preferences = l[index: index + length]
            in_preferences.append(1 - sum(in_preferences))
            index += length
            self.preferences.append(in_preferences)
        return self
    
    def largest_preference(self):
        return np.max(self.preferences)
    
    def smallest_preference(self):
        return np.min(self.preferences)

    def get_road_in(self, road_number):
        return self.roads_in[road_number] if self.roads_in[road_number] else "Empty"

    def get_road_out(self, road_number):
        return self.roads_out[road_number] if self.roads_out[road_number] else "Empty"
    
    def get_roads_in(self): 
        return self.roads_in  
    
    def get_roads_out(self): 
        return self.roads_out  
    
    def get_preferences(self): 
        return self.preferences

    def resolve(self):
        if None in self.roads_in + self.roads_out:
            return "Input/output roads not fully initialized, resolution at junction is incomplete."

        #################################################################
        ### Enhanced Right of Way with Capacity Management
        #################################################################

        c_in = list(map(lambda road_in: road_in.entropy_max_flux_in(), self.roads_in))
        c_out = list(map(lambda road_out: road_out.entropy_max_flux_out(), self.roads_out))

        # Compute flux demands for each outgoing road
        fluxes_out_demand = [0 for _ in range(len(self.roads_out))]
        for i in range(len(self.roads_in)):
            for j in range(len(self.roads_out)):
                fluxes_out_demand[j] += c_in[i] * self.preferences[i][j]

        # Step 1: Check if we need to activate right of way (component-wise violation check)
        # Right of way (ROW) is activated if ANY outgoing demand exceeds its capacity
        row_activated = any(fluxes_out_demand[j] > c_out[j] for j in range(len(self.roads_out)))
        
        if not row_activated:
            # Step 2: Right of way NOT activated - use normal logic
            for i, road in enumerate(self.roads_in):
                flux_temp = c_in[i]
                road.right_flux_from_junction = flux_temp
                road.update_density_in(flux_temp)

            for j, road in enumerate(self.roads_out):
                flux_temp = fluxes_out_demand[j]
                # Only store flux, don't update density
                road.left_flux_from_junction = flux_temp
                road.update_density_out(flux_temp)
        else:
            # Step 3: Right of way IS activated
            total_demand = sum(fluxes_out_demand) # This should be equal to sum(c_in)
            total_capacity = sum(c_out)
            
            if total_demand > total_capacity:
                # Step 4: ROW and total demand > total capacity
                # Scale incoming fluxes by total_capacity/total_demand
                scaling_factor = total_capacity / total_demand
                
                for i, road in enumerate(self.roads_in):
                    flux_temp = c_in[i] * scaling_factor
                    # Only store flux, don't update density
                    road.right_flux_from_junction = flux_temp
                    road.update_density_in(flux_temp)
                
                # Set outgoing fluxes to individual capacities
                for j, road in enumerate(self.roads_out):
                    flux_temp = c_out[j]
                    # Only store flux, don't update density
                    road.left_flux_from_junction = flux_temp
                    road.update_density_out(flux_temp)
            else:
                # Step 5: ROW but total demand <= total capacity
                # Scale outgoing fluxes by total_demand/total_capacity
                scaling_factor = total_demand / total_capacity
                
                for i, road in enumerate(self.roads_in):
                    flux_temp = c_in[i]
                    # Only store flux, don't update density
                    road.right_flux_from_junction = flux_temp
                    road.update_density_in(flux_temp)
                
                for j, road in enumerate(self.roads_out):
                    flux_temp = c_out[j] * scaling_factor
                    # Only store flux, don't update density
                    road.left_flux_from_junction = flux_temp
                    road.update_density_out(flux_temp)