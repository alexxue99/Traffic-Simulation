import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation

class Road(object):
    dt = None
    pj = None


    def __init__(self,name, length = None):
        self.name = name
        self.length = length # This is recorded in miles
        self.lanes = 0
        self.p_j = 0 #jam density
        self.free_flow_speed = 0 # This is recorded in mi/sec (technically normalized by 1/p_j but doesn't affect anything)
        self.sigma = None
        self.cap = 0 # This is the flux capacity in veh/sec/lane (not normalized or scaled by lanes)
        self.max_flux = None # This is recorded in 1/p_j * veh/sec (normalized by p_j, and NOT veh/sec/lane since we scale function based on lanes)
        self.A = None
        self.flux_1 = lambda : None # linear part of piecewise function
        self.flux_2 = lambda : None # quadratic part of piecewise function
        self.flux = lambda: None
        self.max_wave_speed = None
        self.flux_to_rho_m = lambda: None
        self.flux_to_rho_p = lambda: None
        self.cells = None           # We are looking at the location of the cell centers
        self.grid_points = None            # These are the grid points, which are used for flux and boundary computation
        self.init_density = None    # Setting up initial density using data points; assuming cell centers
        self.init_density_func = None # Setting up initial density using a function
        self.current_density = None # This is the current density at the cell centers
        self.current_flux_grid = None # This is the current flux at the grid points, which alternates between the cell centers
        self.left_junction = None   # This should be an instance of the class Junction
        self.right_junction = None  # This should be an instance of the class Junction
        self.left_boundary_function = None # This refers to any custom-made boundary conditions if its not connected to a junction
        # Note: These are first-order boundary functions; left/right boundary_function here takes in the cell right before the boundary cell
        #       and return the value on the boundary.
        self.right_boundary_function = None
        self.density_history = []
        self.is_exit_road = False
        self.dt = 0
        self.dx = 0
        self.time_integrated_cars_exited = 0 # tracks total number of cars that has left this road
        self.time_integrated_cars_on_the_road = 0 # tracks the integral of the number of cars on the road up to time T
        # Since roads are assumed to go from left to right, this tracks the number of cars leaving the road from the right
        self.time_steps_elapsed = 0
        self.direction = "LEFT_TO_RIGHT"
        self.los = None # level of service bounds on density [A,B,C,D,E,F], to be computed later based on free_flow_speed and lanes
        # Option 3: Store junction fluxes directly
        self.left_flux_from_junction = None  # Flux coming from left junction
        self.right_flux_from_junction = None  # Flux going to right junction

    def __eq__(self, other):
        if isinstance(other, Road):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    ##########################################
    ### Setter functions
    ##########################################
    def set_name(self, name):
        self.name = name
        return self

    def set_length(self, length):
        self.length = length
        return self

    def set_lanes(self, lanes):
        self.lanes = lanes
        return self

    def set_free_flow_speed(self, free_flow_speed):
        self.free_flow_speed = free_flow_speed / 3600 # to get in terms of seconds
        return self
    
    def set_cap(self, cap):
        self.cap = cap / 3600 # to get in terms of seconds
        return self

    def set_p_j(self, p_j):
        self.p_j = p_j
        return self

    def set_coefficients(self, f_m, sigma, A):
        if (f_m == None or sigma == None or A == None):
            [f_m, sigma, A] = self.compute_coefficients()
            self.sigma = sigma # value that maximizes the flux
            self.max_flux = f_m
            self.A = A
            return self
        else:
            self.sigma = sigma # value that maximizes the flux
            self.max_flux = f_m
            self.A = A
            return self

    def set_flux_1(self, flux_1):
        self.flux_1 = flux_1
        return self

    def set_flux_2(self, flux_2):
        self.flux_2 = flux_2
        return self

    def set_flux(self, flux):
        self.flux = flux
        return self

    def set_sigma(self, sigma):
        self.sigma = sigma
        return self

    def set_max_wave_speed(self, max_wave_speed):
        if max_wave_speed == None:
            self.max_wave_speed = self.compute_max_wave_speed()
            return self
        else:
            self.max_wave_speed = max_wave_speed
            return self

    def set_invert_flux_m(self, flux_to_rho_m) :
        if abs(flux_to_rho_m(self.flux(self.sigma/2)) - self.sigma/2) > 1e-5:
            raise Exception("Error in flux_to_rho_m")
        self.flux_to_rho_m = flux_to_rho_m
        return self

    def set_invert_flux_p(self, flux_to_rho_p) :
        self.flux_to_rho_p = flux_to_rho_p
        return self

    def set_grid(self, dt):
        if self.max_wave_speed == None:
            raise Exception("You have not initialized the flux function for this road.")
        self.dt = dt
        dx = self.max_wave_speed*dt*2
        nx = int(self.length/dx)
        self.dx = dx
        self.cells = np.linspace(0.0 - 0.5 * dx, self.length + 0.5 * dx, num=nx+2) 
        # These cell centers off the roads are used to track density of ghost cells
        self.grid_points = np.linspace(0.0, self.length, num=nx+1)
        return self

    def set_initial_density_func(self, initial_density_func, scaled = False):
        if self.dt == 0:
            raise Exception("You have not initialized the grid for this road.")
        self.init_density_func = lambda x: initial_density_func(x) * (1 if scaled else self.lanes) #?s

        self.set_initial_density(self.init_density_func(self.cells))
        return self
    
    def set_initial_density(self, initial_density):
        # Check if the size of the np.array matches the size of cells
        if len(initial_density) != len(self.cells):
            raise Exception("Initial density does not match the size of the cells")
        self.init_density = copy.deepcopy(initial_density)
        self.current_density = copy.deepcopy(initial_density)

        self.density_history = [self.current_density,] # This clears the previous density history and input the new initial density in a list.
        return self
    
    def set_zero_initial_density(self):
        # These are for "added/opened" roads, where the initial density is set to 0.
        self.init_density = np.zeros_like(self.cells)


    def set_left_boundary_function(self, left_boundary_function, scaled=False):
        if isinstance(left_boundary_function, str):
            # Handle string cases like "non-reflecting"
            self.left_boundary_function = left_boundary_function
        else:
            # Handle function cases - exactly like set_initial_density_func pattern
            if scaled:
                self.left_boundary_function = left_boundary_function
            else:
                # Create wrapper that scales the result by lanes (not the input)
                self.left_boundary_function = lambda time: left_boundary_function(time, self.get_sigma()) * self.lanes
        return self

    def set_right_boundary_function(self, right_boundary_function, scaled=False):
        if isinstance(right_boundary_function, str):
            # Handle string cases like "non-reflecting"
            self.right_boundary_function = right_boundary_function
        else:
            # Handle function cases - exactly like set_initial_density_func pattern
            if scaled:
                self.right_boundary_function = right_boundary_function
            else:
                # Create wrapper that scales the result by lanes (not the input)
                self.right_boundary_function = lambda time: right_boundary_function(time, self.get_sigma()) * self.lanes
        return self

    def set_is_exit_road(self):
        self.is_exit_road = True
        return self

    def reset_by_func(self):
        # resets cars exited and initial density, and time_steps_elapsed, using initial_density function
        self.time_integrated_cars_exited = 0
        self.time_integrated_cars_on_the_road = 0
        self.time_steps_elapsed = 0
        self.current_density = self.init_density_func(self.cells)
        self.density_history = [self.current_density,]

    def reset_by_data(self):
        # resets cars exited and initial density, and time_steps_elapsed, using initial_density
        self.time_integrated_cars_exited = 0
        self.time_integrated_cars_on_the_road = 0
        self.time_steps_elapsed = 0
        try:
            self.current_density = copy.deepcopy(self.init_density)
            self.density_history = [self.current_density,]
            self.left_flux_from_junction = None
            self.right_flux_from_junction = None
        except:
            raise Exception("Initial density not set. Please set initial density using either initial_density_func or initial_density.")

    def clear_density_history(self):
        # clears the density history
        self.density_history = []

    def set_up(self):
        # sets up flux_1, flux_2, flux, max_wave_speed, invert_m, invert_p, and LOS
        self.set_flux_1(self.compute_flux_1)
        self.set_flux_2(self.compute_flux_2)
        self.set_flux(self.compute_flux)
        self.set_max_wave_speed(None)
        self.set_invert_flux_m(self.compute_inverse_m)
        self.set_invert_flux_p(self.compute_inverse_p)
        self.los = self.compute_los()
        return self

    def set_params(self, speed, lanes, length, cap, init_density_factor=0):
        """
        Sets speed, lanes, length, and capacity as specified.

        The initial density is set to lambda x: init_density_factor * sigma * np.ones_likes(x).
        Does not modify the boundary functions of the road.
        """
        (self.set_length(length)
        .set_lanes(lanes)
        .set_free_flow_speed(speed)
        .set_cap(cap)
        .set_p_j(Road.p_j)
        .set_coefficients(None, None, None)
        .set_up()
        .set_grid(Road.dt)
        .set_initial_density_func(lambda x, sig = self.get_sigma(): init_density_factor*np.ones_like(x)))
        return self

    @staticmethod
    def create_roads(names: list, speeds: list, lanes: list, lengths: list, caps: list, factor, descriptor = "", direction = "LEFT_TO_RIGHT"):
        """
        Creates a list of roads with names, speeds, lanes, lengths, and capacities as specified.

        The initial density of all the roads is set to a constant factor * sigma * np.ones_like(x).
        Does not modify boundary functions or set exit roads either, so those must be manually set as well.

        Returns the list of roads after.
        """
        lists = [names, speeds, lanes, lengths, caps]
        if any(len(lst) != len(lists[0]) for lst in lists):
            raise Exception("Lists don't have same length")
        
        if descriptor != "":
            descriptor = descriptor.lstrip()
        # road names are of the form hwy30 0: prison -> dickenson
        l = [Road(descriptor + " " + str(i) + ": " + names[i]) for i in range(len(names))]
        for i, r in enumerate(l):
            (r.set_length(lengths[i])
            .set_lanes(lanes[i])
            .set_free_flow_speed(speeds[i])
            .set_cap(caps[i])
            .set_p_j(Road.p_j)
            .set_coefficients(None, None, None)
            .set_up()
            .set_grid(Road.dt)
            .set_initial_density_func(lambda x, sig = r.get_sigma(): factor*np.ones_like(x))
            .set_direction(direction))
        return l
    
    def create_roads_flipped(left_roads: list['Road']):
        l = copy.deepcopy(left_roads)
        for road in l:
            road.left_junction = None
            road.right_junction = None
            road.left_boundary_function = None
            road.right_boundary_function = None
            road.name = road.name.replace("->", "<-", 1).replace("left", "right")
            road.set_direction("RIGHT_TO_LEFT")
        return l
    
    def set_direction(self, direction):
        self.direction = direction
        return self

    def set_direction_up(self):
        '''
        Sets direction as down to up.
        '''
        self.direction = "DOWN_TO_UP"
        return self
    
    def set_direction_down(self):
        '''
        Sets direction as up to down.
        '''
        self.direction = "UP_TO_DOWN"
        return self

    def set_direction_left(self):
        '''
        Sets direction as right to left.
        '''
        self.direction = "RIGHT_TO_LEFT"
        return self

        
    ##########################################
    ### Getter functions
    ##########################################

    def get_name(self):
        return self.name

    def get_length(self):
        return self.length

    def get_flux(self):
        return (self.flux, self.sigma, self.max_wave_speed, self.flux_to_rho_m, self.flux_to_rho_p)

    def get_cells(self):
        return self.cells

    def get_max_wave_speed(self):
        return self.max_wave_speed
    
    def get_wave_speed(self,rho):
        """
        Returns the wave speed at a given density rho.
        Assumes that rho is a scalar (not a numpy array).
        """
        if rho <= self.sigma:
            return self.free_flow_speed
        else:
            return 2 * self.A * (rho - self.sigma) 


    def get_max_flux(self):
        return self.max_flux

    def get_free_flow_speed(self):
        return self.free_flow_speed

    def get_current_density(self):
        return copy.deepcopy(self.current_density)
    
    def get_initial_density(self):
        return self.init_density

    def get_left_junction(self):
        return self.left_junction if self.left_junction else "Empty"

    def get_right_junction(self):
        return self.right_junction if self.right_junction else "Empty"

    def get_left_boundary_function(self, left_boundary_function):
        return self.left_boundary_function if self.left_boundary_function else "Empty"

    def get_right_boundary_function(self, right_boundary_function):
        return self.right_boundary_function if self.right_boundary_function else "Empty"

    def get_density_history(self):
        return self.density_history

    def get_total_cars_exited(self):
        return self.time_integrated_cars_exited

    def get_time_integrated_cars_on_the_road(self):
        return self.time_integrated_cars_on_the_road
    
    def get_time_steps_elapsed(self):
        return self.time_steps_elapsed
    
    def get_sigma(self):
        return self.sigma

    def plot_current_density(self):
        fig = plt.figure(figsize=(6.0, 4.0))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\rho$')
        plt.grid()
        plt.ylim([-0.1,self.lanes+0.1])
        plt.xlim([0,self.length])
        plt.title(r'$\rho(x,T)$ along ' + self.name)
        plt.plot(self.cells, self.current_density, color='C0', linestyle='-', linewidth=2)
        plt.show()

    def plot_density_history(self, num_frames=10):
        fig = plt.figure(figsize=(6.0, 4.0))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\rho$')
        plt.grid()
        plt.ylim([-0.1,self.lanes+0.1])
        plt.xlim([0,self.length])
        plt.title(r'$\rho(x,T)$ along ' + self.name)
        for i in range(num_frames):
            frame_no = int(i * len(self.density_history) / num_frames)
            plt.plot(self.cells, self.density_history[frame_no], color='k', linestyle='-', linewidth=2, alpha = 0.4 + 0.6/num_frames*i)
        plt.show()

    def animate_density_history(self, filename):
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['figure.dpi'] = 150  
        fig, ax = plt.subplots()
        d_hist = self.get_density_history()
        x= np.linspace(0, self.length, len(d_hist[0]))
        line, = ax.plot([], [], lw=2)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$\rho$')
        plt.grid()
        plt.ylim([-0.1,self.lanes+0.1])
        plt.xlim([0,self.length])
        plt.title(r'$\rho(x,T)$ along ' + self.name)

        def animate(t):
            line.set_data(x, d_hist[t])
            ax.set_title(r'$\rho(x,T)$ along ' + self.name + f' at timestep {t}')
            return line,

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(d_hist) - 1, blit=False)

        # Explicit writer (prevents the one-frame issue)
        writer = matplotlib.animation.PillowWriter(fps=30)
        ani.save(filename, writer=writer, dpi=200)


    ##########################################
    ### Methods
    ##########################################

    def godunov_flux(self, rhoL, rhoR):
        """
        This is the Riemann solver for the flux function.
        It computes the Godunov flux at the cell boundaries using the exact Riemann solver (in some sense)

        We assume that the input densities are np.arrays of the same size so that this can be vectorized.
        """
        fL = self.flux(rhoL)
        fR = self.flux(rhoR)

        left_smaller = rhoL < rhoR
        # This is also the "straddle" case, with the density at sigma with max flux
        peak_rho = self.sigma # This has already accounted for the number of lanes
        max_flux = self.max_flux # This has already accounted for the number of lanes
        # This is the speed of the shock by Rankine-Hugoniot condition if it exists
        shock_speed = np.divide(fR - fL, rhoR - rhoL, out=np.zeros_like(fR), where=(rhoR != rhoL))  #

        # Case 1: rho_L < rho_R 
        m1 = left_smaller & (rhoR <= peak_rho)   # interval entirely left of peak
        m2 = left_smaller & (rhoL >= peak_rho)   # interval entirely right of peak (a shock but takes fR)
        # Alternatively, the interval straddles the peak.
        # Since rho_L < rho_R, this must be a shock
        m3 = left_smaller & (shock_speed > 0) & ~(m1 | m2)        # interval straddles the peak
        m4 = left_smaller & (shock_speed <= 0) & ~(m1 | m2)       # interval straddles the peak, but shock speed is negative

        # Case 2: rho_L >= rho_R
        m5 = ~left_smaller & (rhoL <= peak_rho)   # both states left of peak
        m6 = ~left_smaller & (rhoR >= peak_rho)   # both states right of peak
        m7 = ~left_smaller & ~(m4 | m5)           # interval straddles the peak
        # Assemble result with one pass of np.select  (seven mutually exclusive masks)
        conditions = [m1, m2, m3, m4, m5, m6, m7]
        # if self.name == "front -> exit hwy30 7":
        #     print(fL, fR, max_flux)
        choices    = [fL, fR, fL, fR, fL, fR, max_flux] # Last element scaled for numerical stability
        return np.select(conditions, choices, default=np.nan)   # default never hit



    def evolve(self):
        """
        This evolves the density on the road internally and on ends of roads that are not connected
        to any other junctions/roads using the numerical boundary conditions. The density connecting
        to the junctions are not to be computed yet, and will be inputted by resolve_junction
        method available in the Junction class.

        To do so, we follow the Godunov scheme with an exact Riemann solver.
        1. Compute the flux at the cell boundaries using a Riemann solver.
        2. Advance the road internally in time using the Godunov scheme.
        """
        self.update_flux_grid() # Updates the flux to be used for resolving the junctions
        F = self.current_flux_grid
        
        # Advance the road internally in time using the Godunov scheme
        temp = self.current_density[1:-1] - self.dt/self.dx * (F[1:] - F[:-1])
        self.current_density[1:-1] = copy.deepcopy(temp)


        ### Recall that this was done after we have updated the internal densities. Any shock/rarefaction
        ###     propagation should have an updated flux grid.
        # self.update_flux_grid()
        
        self.time_integrated_cars_exited += self.current_flux_grid[-1] * self.dt * self.p_j
        # Record number of cars on the road
        # Inner Road; No contributions from the left and right boundary
        self.time_integrated_cars_on_the_road += np.sum(self.current_density[1:-1])*self.dx*self.dt
        self.time_steps_elapsed += 1
        
        # Option 3: Do NOT clear junction fluxes - they persist until junction sets new values
        # This ensures flux conservation between junction resolution calls

    def update_flux_grid(self):
        rhoL = self.current_density[:-1].copy()
        rhoR = self.current_density[1:].copy()
        self.current_flux_grid = self.godunov_flux(rhoL, rhoR)
        
        if self.left_junction is not None and self.left_flux_from_junction is not None:
            self.current_flux_grid[0] = self.left_flux_from_junction
        if self.right_junction is not None and self.right_flux_from_junction is not None:
            self.current_flux_grid[-1] = self.right_flux_from_junction

    def boundary_resolve(self):
        """
        This resolves the boundary conditions for the road, if it is not connected to a junction.
        """
        # Left boundary condition, if it is not attached to a junction
        if self.left_junction == None:
            if self.left_boundary_function == "non-reflecting":
                self.current_density[0] = self.current_density[1]
            else:
                if self.get_wave_speed(self.current_density[1]) < 0:
                    # Essentially a non-reflecting condition even if boundary functions are specified
                    self.current_density[0] = self.current_density[1]  
                    #self.current_density[0] = self.left_boundary_function(self.dt*self.time_steps_elapsed)
                else:
                    self.current_density[0] = self.left_boundary_function(self.dt*self.time_steps_elapsed) 

        # Right boundary condition, if it is not attached to a junction
        if self.right_junction == None:
            if self.right_boundary_function == "non-reflecting":
                self.current_density[-1] = self.current_density[-2] # This is the self-reflecting condition
            else:
                if self.get_wave_speed(self.current_density[-2]) > 0:
                    # Essentially a non-reflecting condition even if boundary functions are specified
                    self.current_density[-1] = self.current_density[-2]
                    #self.current_density[-1] = self.right_boundary_function(self.dt*self.time_steps_elapsed)
                else:
                    self.current_density[-1] = self.right_boundary_function(self.dt*self.time_steps_elapsed)

    def record_road(self):
        return self.density_history.append(self.current_density.copy())

    def evolve_and_record(self):
        """
        This evolves the density on the road internally and on ends of roads that are not connected
        to any other junctions/roads using the numerical boundary conditions. The density connecting
        to the junctions are not to be computed yet, and will be inputted by resolve_junction
        method available in the Junction class.

        Remark: This should ONLY BE USED FOR TESTING PURPOSES/single road (ie not in a junction/network).
        """
        self.evolve()
        self.boundary_resolve()
        self.record_road()


    # computes the max flux under entropy conditions, at a junction where this road is incoming
    def entropy_max_flux_in(self):
        density = self.current_density[-2]
        if density <= self.sigma:
            if self.flux(density) * self.dt >= density:
                return density / self.dt
            else:
                return self.flux(density)
        else:
            return self.max_flux

    # computes the max flux under entropy conditions, at a junction where this road is outgoing
    def entropy_max_flux_out(self):
        density = self.current_density[1]
        if density <= self.sigma:
            return self.max_flux
        else:
            if self.flux(density) * self.dt >= (self.lanes - density):
                return (self.lanes - density) / self.dt
            else:
                return self.flux(density)

    # updates the density, at a junction where this road is incoming
    def update_density_in(self, net_flux):
        self.current_density[-1] = self.flux_to_rho_p(net_flux)

    # updates the density, at a junction where this road is outgoing
    def update_density_out(self, net_flux):
        self.current_density[0] = self.flux_to_rho_m(net_flux)

    # computes the coefficients for the flux function (after normalizing by p_j)
    def compute_coefficients(self):
        f_m = self.cap / self.p_j
        sigma = (f_m / self.free_flow_speed)
        A = -f_m / ((1 - sigma)**2)
        if self.lanes == 1: # now we also scale by number of lanes
            return [f_m, sigma, A]
        else:
            p_new = self.lanes
            f_m_new = f_m * p_new
            sigma_new = sigma * p_new
            A_new = A / p_new
            return [f_m_new, sigma_new, A_new]

    # computes the heaviside flux function
    def compute_flux(self, rho):
        fl_ind_1 = np.heaviside(-(rho- self.sigma), 0.5) #indicates rho values corresponding to flux_1
        fl_ind_2 = np.heaviside(rho- self.sigma, 0.5) #indicates rho values corresponding to flux_2
        return np.multiply(self.flux_1(rho), fl_ind_1)  + np.multiply(self.flux_2(rho), fl_ind_2)# the flux function

    # compute the max wave speed
    def compute_max_wave_speed(self):
        return max(self.free_flow_speed, abs(2 *self.A * (self.lanes - self.sigma)))

    # compute linear portion of flux
    def compute_flux_1(self, rho):
        return rho * self.free_flow_speed

    # compute quadratic portion of flux
    def compute_flux_2(self, rho):
        return self.A * (rho - self.sigma)**2 + self.max_flux

    # compute smaller inverse of flux
    def compute_inverse_m(self, gamma):
        return gamma / self.free_flow_speed

    # compute larger inverse of flux
    def compute_inverse_p(self, gamma):
        return self.sigma + np.sqrt((gamma - self.max_flux)/(self.A))
    
    # compute density from velocity, where v = v_f * alpha
    def compute_rho_from_v(self, alpha):
        v = alpha * self.free_flow_speed
        return self.sigma + (v / (2 * self.A)) + np.sqrt((self.max_flux * (alpha - 1) / self.A) + ((v**2) / (4 * self.A**2)))
    
    @staticmethod
    def reset_junctions(road_list: list['Road']):
        """
        Resets the left and right junction of every road in road_list to be None.
        """
        for r in road_list:
            r.left_junction = None
            r.right_junction = None

    def compute_los(self):
        #LOS A: free flow, up until sigma
        #los_A = self.sigma
        los_A = 0.5 * self.sigma
        #LOS B: stable flow, up until 0.7 * lanes
        los_B = self.compute_rho_from_v(0.7).item()
        #LOS C: unstable flow, up until 0.5 * lanes
        los_C = self.compute_rho_from_v(0.5).item()
        #LOS D: congested flow, up until 0.4 * lanes
        los_D = self.compute_rho_from_v(0.4).item()
        #LOS E: very congested flow, up until 0.3 * lanes
        los_E = self.compute_rho_from_v(0.3).item()
        #LOS F: jam density, anything above los_E
        los_F = self.lanes
        return [los_A, los_B, los_C, los_D, los_E, los_F]
    
    def compute_los_level(self, density):
        # A - 0, B - 1, C - 2, D - 3, E - 4, F - 5
        prev = 0
        for i,d in enumerate(self.los):
            if d >= density:
                return max(0, (density - prev) / (d - prev) + i - 1)
            prev = d
        return 5 # level F

    def compute_average_density(self):
        """
        Computes the average density of the road.
        """
        return np.mean(self.current_density[1:-1])
    
    def compute_average_los(self):
        """
        Computes the average level of service of the road.
        """
        avg_density = self.compute_average_density()
        return self.compute_los_level(avg_density)
