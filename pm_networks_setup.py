from roads_setup import *

from junction import Junction
from network import Network
import copy

### Special roads and road list for PM network

def create_pm_special_roads(hwy30, bypass, front, wainee):
    hwy30_down = Road.create_roads_flipped(hwy30[0:6])
    hwy30_down[0].set_is_exit_road()
    hwy30_down[0].set_right_boundary_function("non-reflecting")

    bypass_down_list = Road.create_roads_flipped(bypass[0:1])
    bypass_down = bypass_down_list[0]
    bypass_down.set_is_exit_road()
    bypass_down.set_right_boundary_function("non-reflecting")

    front_down = Road.create_roads_flipped(front[0:10])
    front_down[0].set_is_exit_road()
    front_down[0].set_right_boundary_function("non-reflecting")

    wainee_down = Road.create_roads_flipped(wainee[0:8])
    wainee_down[0].set_is_exit_road()
    wainee_down[0].set_right_boundary_function("non-reflecting")

    return hwy30_down, bypass_down, front_down, wainee_down


def setup_pm_networks(gamma, gamma2=None):
    """Create and return pm networks based on the given gamma. gamma2 controls the urgency modification of some sources."""
    all_roads = create_all_roads(gamma)

    # Unpack main groups so the roads are accessible like before
    hwy30, luna_left, luna_right, front, wainee, keawe, bypass, \
    kenui_left, kenui_right, papalaua_left, papalaua_right, \
    dickenson_left, dickenson_right, prison_left, prison_right, \
    kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad, \
    gateway, oilroad, puanoa_pl, baker_left, wahie, canal, \
    baker_right, panaewa, hale, kahoma_village, luakini_up, luakini_down = all_roads
    
    hwy30_down, bypass_down, front_down, wainee_down = create_pm_special_roads(hwy30, bypass, front, wainee)

    road_list_pm = (hwy30 + hwy30_down
                + luna_left[2:-1] + luna_right[2:-1] + [luna_right[0]]
                + front + front_down + wainee + wainee_down
                + bypass + keawe
                + kenui_left + kenui_right
                + [papalaua_right[0]]
                + [puanoa_pl, kahoma_village, baker_left, baker_right, wahie]
                + [komomai, kuhua, pauoa, kale, paunau_st, kalena]
                + [luakini_down, luakini_up, hale, canal, panaewa]
                + dickenson_left + dickenson_right
                + prison_left + prison_right
                + [gateway, oilroad, dirtroad, bypass_down])
    
    sources = [kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad]
    sources_default = [puanoa_pl, baker_left, wahie, canal, baker_right, panaewa,
                    hale, kahoma_village, luakini_up, luakini_down]
    half_sources = [bypass[0], puanoa_pl, kahoma_village, baker_left, baker_right, wahie, front[0], wainee[0], hwy30[0],
                    luakini_down, luakini_up, hale, canal, panaewa]

    ### Creating junctions

    Road.reset_junctions(road_list_pm)

    ### HWY 30
    h1p = Junction("h1") # is exit junction
    h1p.set_roads_in(hwy30[0], hwy30_down[1], prison_left[1]).set_roads_out(hwy30_down[0], hwy30[1], prison_right[1])

    h2p = Junction("h2")
    h2p.set_roads_in(hwy30[1], hwy30_down[2], dickenson_left[2]).set_roads_out(hwy30_down[1], hwy30[2], dickenson_right[2])

    h3p = Junction("h3")
    h3p.set_roads_in(hwy30[2], hwy30_down[3], luna_right[2]).set_roads_out(hwy30_down[2], hwy30[3], luna_left[2])

    h4p = Junction("h4")
    h4p.set_roads_in(hwy30[3], hwy30_down[4]).set_roads_out(hwy30_down[3], hwy30[4])

    h5p = Junction("h5")
    h5p.set_roads_in(hwy30[4], hwy30_down[5], kenui_left[2]).set_roads_out(hwy30_down[4], hwy30[5], kenui_right[2])

    h6p = Junction("h6")
    h6p.set_roads_in(hwy30[5], keawe[0]).set_roads_out(hwy30_down[5], hwy30[6])

    h7p = Junction("h7") # is exit junction
    h7p.set_roads_in(hwy30[6], front[9]).set_roads_out(hwy30[7], front_down[9])


    ### FRONT
    f1p = Junction("f1") #is exit junction
    f1p.set_roads_in(front[0], prison_right[0], front_down[1]).set_roads_out(front[1], prison_left[0], front_down[0])

    f2p = Junction("f2")
    f2p.set_roads_in(front[1], canal, front_down[2]).set_roads_out(front[2], front_down[1])

    f3p = Junction("f3")
    f3p.set_roads_in(front[2], dickenson_right[0], front_down[3]).set_roads_out(front[3], dickenson_left[0], front_down[2])

    f4p = Junction("f4")
    f4p.set_roads_in(front[3], luna_right[0], front_down[4]).set_roads_out(front[4], front_down[3])

    f5p = Junction("f5")
    f5p.set_roads_in(front[4], wahie, front_down[5]).set_roads_out(front[5], front_down[4])

    f6p = Junction("f6")
    f6p.set_roads_in(front[5], papalaua_right[0], front_down[6]).set_roads_out(front[6],front_down[5])

    f7p = Junction("f7")
    f7p.set_roads_in(front[6], baker_left, front_down[7]).set_roads_out(front[7], front_down[6])

    f8p = Junction("f8")
    f8p.set_roads_in(front[7], kenui_right[0], front_down[8]).set_roads_out(front[8], kenui_left[0], front_down[7])

    f9p = Junction("f9")
    f9p.set_roads_in(front[8], puanoa_pl, front_down[9]).set_roads_out(front[9], front_down[8])


    ### WAINEE
    w1p = Junction("w1") # is exit jucntion
    w1p.set_roads_in(wainee[0], prison_left[0], prison_right[1], wainee_down[1]).set_roads_out(wainee[1], prison_right[0], prison_left[1], wainee_down[0])

    w2p = Junction("w2")
    w2p.set_roads_in(wainee[1], hale, wainee_down[2]).set_roads_out(wainee[2], wainee_down[1])

    w3p = Junction("w3")
    w3p.set_roads_in(wainee[2], dickenson_left[1], dickenson_right[2], wainee_down[3]).set_roads_out(wainee[3], dickenson_right[1], dickenson_left[2], wainee_down[2])

    w4p = Junction("w4")
    w4p.set_roads_in(wainee[3], panaewa, wainee_down[4]).set_roads_out(wainee[4], wainee_down[3])

    w5p = Junction("w5")
    w5p.set_roads_in(wainee[4], wainee_down[5]).set_roads_out(wainee[5], luna_right[0], wainee_down[4])

    w6p = Junction("w6")
    w6p.set_roads_in(wainee[5], wainee_down[6]).set_roads_out(wainee[6], papalaua_right[0], wainee_down[5])

    w7p = Junction("w7")
    w7p.set_roads_in(wainee[6], baker_right, wainee_down[7]).set_roads_out(wainee[7], wainee_down[6])

    w8p = Junction("w8")
    w8p.set_roads_in(wainee[7], kenui_left[1], kenui_right[2]).set_roads_out(kenui_right[1], kenui_left[2], wainee_down[7])


    ### LAHAINALUNA
    l1p = Junction("l1")
    l1p.set_roads_in(luna_left[2], kuhua, luna_right[3]).set_roads_out(luna_right[2], luna_left[3])

    l2p = Junction("l2")
    l2p.set_roads_in(luna_left[3], pauoa, luna_right[4]).set_roads_out(luna_right[3], luna_left[4])

    l3p = Junction("l3")
    l3p.set_roads_in(luna_left[4], kale, luna_right[5]).set_roads_out(luna_right[4], luna_left[5])

    l4p = Junction("l4")
    l4p.set_roads_in(luna_left[5], paunau_st, luna_right[6]).set_roads_out(luna_right[5], luna_left[6])

    # l5p = Junction("l5")
    # l5p.set_roads_in(luna_left[6], luna_right[7]).set_roads_out(luna_right[6], luna_left[7])

    l5p_left = Junction("l5_left")
    l5p_left.set_roads_in(luna_left[6]).set_roads_out(luna_left[7])

    l5p_right = Junction("l5_right")
    l5p_right.set_roads_in(luna_right[7]).set_roads_out(luna_right[6])

    l6p = Junction("l6")
    l6p.set_roads_in(luna_left[7], kalena, luna_right[8]).set_roads_out(luna_right[7], luna_left[8])

    l7p = Junction("l7")
    l7p.set_roads_in(luna_left[8], luna_right[9], dirtroad).set_roads_out(luna_right[8], luna_left[9])

    l8p = Junction("l8") #is exit junction
    l8p.set_roads_in(luna_left[9], bypass[0]).set_roads_out(luna_right[9], bypass[1], bypass_down)

    ### KEAWE
    k1p = Junction("k1")
    k1p.set_roads_in(keawe[1]).set_roads_out(keawe[0], gateway)

    k2p = Junction("k2")
    k2p.set_roads_in(keawe[2], komomai).set_roads_out(keawe[1], oilroad)

    k3p = Junction("k3")
    k3p.set_roads_in(bypass[1]).set_roads_out(keawe[2])

    ### KENUI - KAHOMA
    kenui_kahoma_p = Junction("kenui-kahoma")
    kenui_kahoma_p.set_roads_in(kenui_left[0], kenui_right[1], kahoma_village).set_roads_out(kenui_right[0], kenui_left[1])


    ### Dickenson
    d1p = Junction("d1")
    d1p.set_roads_in(dickenson_left[0], dickenson_right[1], luakini_down, luakini_up).set_roads_out(dickenson_right[0], dickenson_left[1])

    ### Creating network
    __network_pm = Network([h1p, h2p, h3p, h4p, h5p, h6p, h7p, f1p, f2p, f3p, f4p, f5p, f6p, f7p, f8p, f9p, w1p, w2p, w3p, w4p, w5p, w6p, w7p, w8p,
                    l1p, l2p, l3p, l4p, l5p_left, l5p_right, l6p, l7p, l8p,
                    k1p, k2p, k3p, kenui_kahoma_p, d1p])

    __network_pm.check_num_junctions(38)
    __network_pm.check_roads_in_roadlist(road_list_pm)

    network_pm_base = copy.deepcopy(__network_pm)
    road_list_pm_base = copy.deepcopy(road_list_pm)

    to_remove = [gateway, oilroad, dirtroad, bypass_down] + hwy30_down + front_down + wainee_down

    for r in to_remove:
        road_list_pm_base.remove(r)
        network_pm_base.remove_road(r)

    network_pm_base.modify_urgency(half_sources, gamma)
    network_pm_base.check_roads_in_roadlist(road_list_pm_base)

    ## By default, h7p is the exit road, and needs to be run once to initialize the distances
    network_pm_base.compute_distances(h7p)  

    network_pm_2 = copy.deepcopy(__network_pm)
    road_list_pm_2 = copy.deepcopy(road_list_pm)

    to_remove = [bypass[1], keawe[2], oilroad, dirtroad, bypass_down] + hwy30_down + front_down + wainee_down

    for r in to_remove:
        road_list_pm_2.remove(r)
        network_pm_2.remove_road(r)

    network_pm_2.modify_urgency(sources + sources_default + half_sources, 1 if not gamma2 else gamma2)
    network_pm_2.check_roads_in_roadlist(road_list_pm_2)

    ## By default, h7p is the exit road, and needs to be run once to initialize the distances
    network_pm_2.compute_distances(h7p)  

    network_pm_3 = copy.deepcopy(__network_pm)
    road_list_pm_3 = copy.deepcopy(road_list_pm)

    to_remove = [bypass[1], keawe[2], dirtroad, bypass_down] + hwy30_down + front_down + wainee_down

    for r in to_remove:
        road_list_pm_3.remove(r)
        network_pm_3.remove_road(r)

    network_pm_3.modify_urgency(sources + sources_default + half_sources, 1 if not gamma2 else gamma2)
    network_pm_3.check_roads_in_roadlist(road_list_pm_3)

    ## By default, h7p is the exit road, and we need to initialize the distances to it
    network_pm_3.compute_distances(h7p)  

    network_pm_4 = copy.deepcopy(__network_pm)
    road_list_pm_4 = copy.deepcopy(road_list_pm) 

    to_remove = [bypass[1], keawe[2], dickenson_left[1], dickenson_right[1], dirtroad, bypass_down] + hwy30_down + front_down + wainee_down

    for r in to_remove:
        road_list_pm_4.remove(r)
        network_pm_4.remove_road(r)

    network_pm_4.modify_urgency(sources + sources_default + half_sources, 1 if not gamma2 else gamma2)
    network_pm_4.check_roads_in_roadlist(road_list_pm_4)

    ## By default, h5p is the exit road, and needs to be run once to initialize the distances
    network_pm_4.compute_distances(h7p)  

    network_pm_5 = copy.deepcopy(__network_pm)
    road_list_pm_5 = copy.deepcopy(road_list_pm)

    to_remove = [bypass[1], keawe[2], dickenson_left[1], dickenson_right[1]]
    for r in to_remove:
        road_list_pm_5.remove(r)
        network_pm_5.remove_road(r)

    network_pm_5.modify_urgency(sources + sources_default + half_sources, 1 if not gamma2 else gamma2)
    network_pm_5.modify_urgency([hwy30[0], bypass[0]], 1/2)
    network_pm_5.check_roads_in_roadlist(road_list_pm_5)
    network_pm_5.compute_distances(h7p)  

    exits = [h7p, h1p, f1p, w1p, l8p] # new exit junctions 
    network_pm_5.compute_distances_mult_exits(exits)  

    network_pm_2.junctions.remove(k3p)
    network_pm_3.junctions.remove(k3p)
    network_pm_4.junctions.remove(k3p)
    network_pm_5.junctions.remove(k3p)

    network_pm_base.update_roads() 
    network_pm_2.update_roads()
    network_pm_3.update_roads()
    network_pm_4.update_roads()
    network_pm_5.update_roads()

    return network_pm_base, network_pm_2, network_pm_3, network_pm_4, network_pm_5