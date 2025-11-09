##############################
###  Preamble
##############################
import numpy as np
from road import Road

##############################
###  Roads
##############################
### Creating all roads
#print(gamma)
Road.dt = 0.1       # in seconds
Road.p_j = 200      # 223 pc/km/lane ~> 357 pc/mi/lane # less veh/mi/lane than pc's, use default value 200 
DEFAULT = 0.01 # default length of a source road

def create_all_roads(gamma):
    """Wrap everything in a function so that the gamma dependence is explicit
    Returns: all_roads = [
    hwy30, luna_left, luna_right, front, wainee, keawe, bypass,
    kenui_left, kenui_right, papalaua_left, papalaua_right,
    dickenson_left, dickenson_right, prison_left, prison_right,
    kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad,
    gateway, oilroad, puanoa_pl, baker_left, wahie, canal,
    baker_right, panaewa, hale, kahoma_village, luakini_up, luakini_down]
    return all_roads"""
    ###################
    ## Hwy-30
    ###################

    # hwy30[7]
    # hwy30[6]
    # \vdots
    # hwy30[0] 

    hwy30_names = ["source -> prison", "prison -> dickenson", "dickenson -> lahainaluna",
                "lauhainaluna -> papalaua", "papalaua -> kenui", "kenui -> keawe",
                "keawe -> front", "front -> exit"]
    hwy30_speeds = [35, 35, 35, 40, 40, 40, 40, 40]
    hwy30_lanes = [2 for _ in range(8)]
    hwy30_lengths = [DEFAULT, 0.28, 0.16, 0.12, 0.32, 0.17, 0.66, DEFAULT]
    hwy30_caps = [875, 875, 875, 1000, 1000, 1000, 1000, 1000]

    hwy30 = Road.create_roads(hwy30_names, hwy30_speeds, hwy30_lanes, hwy30_lengths, hwy30_caps, gamma, "hwy30", "DOWN_TO_UP")

    hwy30[7].set_is_exit_road()
    hwy30[7].set_right_boundary_function("non-reflecting")

    ###################
    ## Lahainaluna Rd
    ###################

    # luna_left[0] -> luna_left[1] -> ... -> luna_left[10] (left to right)
    # luna_right[0] <- luna_right[1] <- ... <- luna_right[10] (right to left)

    luna_names = ["front -> wainee", "wainee -> hwy30", "hwy30 -> kuhua", "kuhua -> pauoa",
                "pauoa -> kale", "kale -> paunau", "paunau -> kelawea",
                "kelawea -> kalena", "kalena -> dirt road",
                "dirt road -> bypass", "bypass -> source"]
    luna_speeds = [20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30]
    luna_lanes = [1 for _ in range(11)]
    luna_lengths = [0.14, 0.093, 0.14, 0.05, 0.09, 0.08, 0.06, 0.12, 0.13, 0.03, DEFAULT]
    luna_caps = [500, 500, 500, 500, 500, 500, 500, 600, 600, 600, 600]

    luna_left = Road.create_roads(luna_names, luna_speeds, luna_lanes, luna_lengths, luna_caps, gamma, "lahainaluna left")
    luna_right = Road.create_roads_flipped(luna_left)

    ###################
    ## Front Street
    ###################

    front_names = ["source -> prison", "prison -> canal", "canal -> dickenson", "dickenson -> lahainaluna",
                "lahainaluna -> wahie", "wahie -> papalaua", "papalaua -> baker",
                "baker -> kenui", "kenui -> puanoa", "puanoa -> hwy30"]
    front_speeds = [20 for _ in range(10)]
    front_lanes = [1 for _ in range(10)]
    front_lengths = [DEFAULT, 0.06, 0.14, 0.16, 0.05, 0.10, 0.17, 0.17, 0.10, 0.78]
    front_caps = [500 for _ in range(10)]

    front = Road.create_roads(front_names, front_speeds, front_lanes, front_lengths, front_caps, gamma, "front", "DOWN_TO_UP")

    ###################
    ## Wainee Street
    ###################
    wainee_names = ["source -> prison", "prison -> hale", "hale -> dickenson", 
                    "dickenson -> panaewa", "panaewa -> lahainaluna", "lahainaluna -> papalaua",
                    "papalaua -> baker", "baker -> kenui"]
    wainee_speeds = [20 for _ in range(8)]
    wainee_lanes = [1 for _ in range(8)]
    wainee_lengths = [DEFAULT, 0.14, 0.10, 0.11, 0.05, 0.14, 0.16, 0.16]
    wainee_caps = [300, 300, 300, 300, 300, 400, 400, 400]

    wainee = Road.create_roads(wainee_names, wainee_speeds, wainee_lanes, wainee_lengths, wainee_caps, gamma, "wainee", "DOWN_TO_UP")
        
    ###################
    ## Keawe
    ###################

    keawe_names = ["hwy30 -> gateway", "gateway -> oil road", "oil road -> bypass"]
    keawe_speeds = [25 for _ in range(3)]
    keawe_lanes = [2, 2, 1]
    keawe_lengths = [0.10, 0.09, 0.36]
    keawe_caps = [550 for _ in range(3)]

    keawe = Road.create_roads(keawe_names, keawe_speeds, keawe_lanes, keawe_lengths, keawe_caps, gamma, "keawe", "RIGHT_TO_LEFT")

    ###################
    ## Lahaina Bypass
    ###################

    bypass_names = ["source -> lahainaluna", "lahainaluna -> keawe"]
    bypass_speeds = [30, 30]
    bypass_lanes = [1, 1]
    bypass_lengths = [DEFAULT, 0.7]
    bypass_caps = [650, 650]

    bypass = Road.create_roads(bypass_names, bypass_speeds, bypass_lanes, bypass_lengths, bypass_caps, gamma, "bypass", "DOWN_TO_UP")

    ###################
    ## Sources/Exits
    ###################

    kuhua = Road("kuhua")
    pauoa = Road("pauoa")
    kale = Road("kale")
    paunau_st = Road("paunau")
    komomai = Road("komo mai")
    kalena = Road("kalena")

    dirtroad = Road("dirt road")

    gateway = Road("gateway")
    oilroad = Road("oil road")
    sources_lengths = [0.28, 0.18, 0.18, 0.18, 0.18, 0.15, 0.18, 0.25]

    puanoa_pl = Road("puanoa")
    baker_left = Road("baker left")
    wahie = Road("wahie")
    canal = Road("canal")
    baker_right = Road("baker right")
    panaewa = Road("panaewa")
    hale = Road("hale")
    kahoma_village = Road("kahoma village way")
    luakini_up = Road("luakini up")
    luakini_down = Road("luakini down")

    sources = [kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad]
    sources_default = [puanoa_pl, baker_left, wahie, canal, baker_right, panaewa,
                    hale, kahoma_village, luakini_up, luakini_down]

    for i, r in enumerate(sources):
        r.set_params(20, 1, sources_lengths[i], 300)

    for r in sources_default + [gateway, oilroad]:
        r.set_params(20, 1, DEFAULT, 300)

    oilroad.set_is_exit_road()
    oilroad.set_right_boundary_function("non-reflecting")

    gateway.set_is_exit_road()
    gateway.set_right_boundary_function(lambda time: 1) # Boundary is one for all time

    for r in [kuhua, kale, kalena, kahoma_village, luakini_down]:
        r.set_direction_down()

    for r in [pauoa, paunau_st, komomai, luakini_up]:
        r.set_direction_up()

    for r in [baker_left, wahie, hale, panaewa, puanoa_pl]:
        r.set_direction_left()

        
    ###################
    ## Kenui Street
    ###################

    kenui_names = ["front -> kahoma village", "kahoma village -> wainee", "wainee -> hwy30"]
    kenui_speeds = [20 for _ in range(3)]
    kenui_lanes = [1 for _ in range(3)]
    kenui_lengths = [0.10, 0.08, 0.02]
    kenui_caps = [400 for _ in range(3)]

    kenui_left = Road.create_roads(kenui_names, kenui_speeds, kenui_lanes, kenui_lengths, kenui_caps, gamma, "kenui left")
    kenui_right = Road.create_roads_flipped(kenui_left)

    ###################
    ## Papalaua Street
    ###################

    papalaua_names = ["front -> wainee", "wainee -> hwy30"]
    papalaua_speeds = [20, 20]
    papalaua_lanes = [1, 1]
    papalaua_lengths = [0.15, 0.07]
    papalaua_caps = [500, 500]

    papalaua_left = Road.create_roads(papalaua_names, papalaua_speeds, papalaua_lanes, papalaua_lengths, papalaua_caps, gamma, "papalaua left")
    papalaua_right = Road.create_roads_flipped(papalaua_left)

    ###################
    ## Dickenson Street
    ###################

    dickenson_names = ["front -> luakini", "luakini -> wainee", "wainee -> hwy30"]
    dickenson_speeds = [20 for _ in range(3)]
    dickenson_lanes = [1 for _ in range(3)]
    dickenson_lengths = [0.05, 0.09, 0.11]
    dickenson_caps = [400 for _ in range(3)]

    dickenson_left = Road.create_roads(dickenson_names, dickenson_speeds, dickenson_lanes, dickenson_lengths, dickenson_caps, gamma, "dickenson left")
    dickenson_right = Road.create_roads_flipped(dickenson_left)

    ###################
    ## Prison Street
    ###################

    prison_names = ["front -> wainee", "wainee -> hwy30"]
    prison_speeds = [20, 20]
    prison_lanes = [1, 1]
    prison_lengths = [0.16, 0.08]
    prison_caps = [300, 300]

    prison_left = Road.create_roads(prison_names, prison_speeds, prison_lanes, prison_lengths, prison_caps, gamma, "prison left")
    prison_right = Road.create_roads_flipped(prison_left)

    ## setting inital densities based on AADT values

    ## Hwy-30 initial densities
    hwy30_los = [None] * len(hwy30)
    for i in range(len(hwy30)):
        hwy30_los[i] = hwy30[i].compute_los()

    hwy_30_los_index = [1,1,1,1,1,1,2,1] # what level of service each road is initialized at (LOS A = 0, B = 1,...)
    for i in range(len(hwy30)):
        j = hwy_30_los_index[i]
        rho_0 = hwy30_los[i][j]
        hwy30[i].set_initial_density_func(lambda x, rho = rho_0: rho*np.ones_like(x), True)

    ## Lahainaluna initial densities
    luna_left_los = [None] * len(luna_left)
    luna_right_los = [None] * len(luna_right)
    for i in range(len(luna_left)):
        luna_left_los[i] = luna_left[i].compute_los()
        luna_right_los[i] = luna_right[i].compute_los()

    luna_los_index = [2,2,2,2,2,2,2,1,1,1,1] # what level of service each road is initialized at (LOS A = 0, B = 1,...)
    for i in range(len(luna_left)):
        j = luna_los_index[i]
        rho_0_left = luna_left_los[i][j]
        rho_0_right = luna_right_los[i][j]
        luna_left[i].set_initial_density_func(lambda x, rho = rho_0_left : rho*np.ones_like(x), True)
        luna_right[i].set_initial_density_func(lambda x, rho = rho_0_right: rho*np.ones_like(x), True)

    ## Front Street initial densities
    front_los = [road.compute_los() for road in front]  #they're all the same
    rho_0 = front_los[1][3]
    for i in range(10):
        front[i].set_initial_density_func(lambda x, rho = rho_0: rho*np.ones_like(x), True)

    ## Wainee Street initial densities
    for i in range(5):
        wainee[i].set_initial_density_func(lambda x: 0*np.ones_like(x), True)

    # it is LOS A, so we don't use the bounds from the function
    for i in range(5,8):
        wainee[i].set_initial_density_func(lambda x: 0.0561*np.ones_like(x), True)

    ## Keawe Street initial densities
    keawe_los_index = [2,1,2]
    keawe_los = [None] * len(keawe)
    for i in range(len(keawe)):
        keawe_los[i] = keawe[i].compute_los()

    for i in range(3):
        j = keawe_los_index[i]
        rho_0 = keawe_los[i][j]
        keawe[i].set_initial_density_func(lambda x, rho = rho_0: rho*np.ones_like(x), True)


    ## Lahaina Bypass initial densities
    bypass_los = [road.compute_los() for road in bypass]
    rho_0 = bypass_los[1][1]
    for i in range(2):
        bypass[i].set_initial_density_func(lambda x, rho = rho_0: rho*np.ones_like(x), True)

    ###### initial densities for minor roads
    ## Kenui Street initial densities -all the same, LOS A
    for i in range(3):
        kenui_left[i].set_initial_density_func(lambda x: 0.0378*np.ones_like(x), True)
        kenui_right[i].set_initial_density_func(lambda x: 0.0378*np.ones_like(x), True)

    ## Papalaua initial densities
    for i in range(2):
        papalaua_left[i].set_initial_density_func(lambda x: 0.0489*np.ones_like(x), True)
        papalaua_right[i].set_initial_density_func(lambda x: 0.0489*np.ones_like(x), True)

    ## Dickenson Street initial densities
    for i in range(3):  
        dickenson_left[i].set_initial_density_func(lambda x: 0.0475*np.ones_like(x), True)
        dickenson_right[i].set_initial_density_func(lambda x: 0.0475*np.ones_like(x), True)

    ## Prison Street initial densities
    for i in range(2):
        prison_left[i].set_initial_density_func(lambda x: 0*np.ones_like(x), True)
        prison_right[i].set_initial_density_func(lambda x: 0*np.ones_like(x), True)


    ### Setting sources' boundary functions and initial densities

    gamma_sources = [kuhua, kale, kalena, pauoa, paunau_st, komomai, dirtroad]
    half_sources = [bypass[0], puanoa_pl, kahoma_village, baker_left, baker_right, wahie, front[0], wainee[0], hwy30[0],
                    luakini_down, luakini_up, hale, canal, panaewa]
    full_sources = [luna_right[-1]]

    for g in gamma_sources:
        g_sigma = g.get_sigma()
        g.set_left_boundary_function(lambda time,sig = g_sigma: gamma)
        g.set_initial_density_func(lambda x, sig = g_sigma: gamma*np.ones_like(x))

    for s in half_sources:
        s_sigma = s.get_sigma()
        s.set_left_boundary_function(lambda time,sig = s_sigma: gamma)
        s.set_initial_density_func(lambda x, sig = s_sigma: gamma*np.ones_like(x))

    for f in full_sources:
        f_sigma = f.get_sigma()
        f.set_left_boundary_function(lambda time,sig = f_sigma: gamma)
        f.set_initial_density_func(lambda x, sig = f_sigma: gamma*np.ones_like(x))

    all_roads = [
    hwy30, luna_left, luna_right, front, wainee, keawe, bypass,
    kenui_left, kenui_right, papalaua_left, papalaua_right,
    dickenson_left, dickenson_right, prison_left, prison_right,
    kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad,
    gateway, oilroad, puanoa_pl, baker_left, wahie, canal,
    baker_right, panaewa, hale, kahoma_village, luakini_up, luakini_down]
    return all_roads

def create_am_sources(gamma):
    hwy30_source = Road("hwy30 source -> lahainaluna")
    hwy30_source.set_params(35, 2, DEFAULT, 875, gamma).set_left_boundary_function(lambda time,sig = hwy30_source.get_sigma(): gamma)

    front_source = Road("front source -> lahainaluna")
    front_source.set_params(20, 1, DEFAULT, 500, gamma).set_left_boundary_function(lambda time,sig = front_source.get_sigma(): gamma)

    wainee_source = Road("wainee source -> lahainaluna")
    wainee_source.set_params(20, 1, DEFAULT, 300, gamma).set_left_boundary_function(lambda time,sig = wainee_source.get_sigma(): gamma)
    return [hwy30_source, front_source, wainee_source]