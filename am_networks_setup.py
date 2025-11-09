from roads_setup import *
from roads_setup import create_all_roads

from junction import Junction
from network import Network
import copy

def create_am_sources(gamma):
    hwy30_source = Road("hwy30 source -> lahainaluna")
    hwy30_source.set_params(35, 2, 0.45, 875, gamma).set_left_boundary_function(lambda time,sig = hwy30_source.get_sigma(): gamma)

    front_source = Road("front source -> lahainaluna")
    front_source.set_params(20, 1, 0.37, 500, gamma).set_left_boundary_function(lambda time,sig = front_source.get_sigma(): gamma)

    wainee_source = Road("wainee source -> lahainaluna")
    wainee_source.set_params(20, 1, 0.41, 300, gamma).set_left_boundary_function(lambda time,sig = wainee_source.get_sigma(): gamma)

    return hwy30_source, front_source, wainee_source

def setup_am_networks(gamma):
    """Create and return am networks based on the given gamma."""
    # we first need to define all the roads that got created in roads_setup
    all_roads = create_all_roads(gamma)

    # Unpack main groups so the roads are accessible like before
    hwy30, luna_left, luna_right, front, wainee, keawe, bypass, \
    kenui_left, kenui_right, papalaua_left, papalaua_right, \
    dickenson_left, dickenson_right, prison_left, prison_right, \
    kuhua, pauoa, kale, paunau_st, komomai, kalena, dirtroad, \
    gateway, oilroad, puanoa_pl, baker_left, wahie, canal, \
    baker_right, panaewa, hale, kahoma_village, luakini_up, luakini_down = all_roads

    ##############################
    ###  AM Networks
    ##############################

    ### Special roads and road list for AM network
    hwy30_source, front_source, wainee_source = create_am_sources(gamma)

    road_list_am = (hwy30[3:] + [hwy30_source]
                + luna_left[0:-1] + luna_right
                + front[4:] + [front_source] + wainee[5:] + [wainee_source]
                + bypass + keawe
                + kenui_left + kenui_right
                + papalaua_left + papalaua_right
                + [puanoa_pl, kahoma_village, baker_left, baker_right, wahie]
                + [komomai, kuhua, pauoa, kale, paunau_st, kalena])

    ### Creating junctions

    Road.reset_junctions(road_list_am)

    ### HWY 30
    h1 = Junction("h1")
    h1.set_roads_in(hwy30_source, luna_left[1], luna_right[2]).set_roads_out(hwy30[3], luna_left[2], luna_right[1])

    h2 = Junction("h2")
    h2.set_roads_in(hwy30[3], papalaua_left[1]).set_roads_out(hwy30[4], papalaua_right[1])

    h3 = Junction("h3")
    h3.set_roads_in(hwy30[4], kenui_left[2]).set_roads_out(hwy30[5], kenui_right[2])

    h4 = Junction("h4")
    h4.set_roads_in(hwy30[5], keawe[0]).set_roads_out(hwy30[6])

    h5 = Junction("h5")
    h5.set_roads_in(hwy30[6], front[9]).set_roads_out(hwy30[7])

    ### FRONT
    f1 = Junction("f1")
    f1.set_roads_in(front_source, luna_right[0]).set_roads_out(front[4], luna_left[0])

    f2 = Junction("f2")
    f2.set_roads_in(front[4], wahie).set_roads_out(front[5])

    f3 = Junction("f3")
    f3.set_roads_in(front[5], papalaua_right[0]).set_roads_out(front[6], papalaua_left[0])

    f4 = Junction("f4")
    f4.set_roads_in(front[6], baker_left).set_roads_out(front[7])

    f5 = Junction("f5")
    f5.set_roads_in(front[7], kenui_right[0]).set_roads_out(front[8], kenui_left[0])

    f6 = Junction("f6")
    f6.set_roads_in(front[8], puanoa_pl).set_roads_out(front[9])


    ### WAINEE
    w1 = Junction("w1")
    w1.set_roads_in(wainee_source, luna_left[0], luna_right[1]).set_roads_out(wainee[5], luna_right[0], luna_left[1])

    w2 = Junction("w2")
    w2.set_roads_in(wainee[5], papalaua_left[0], papalaua_right[1]).set_roads_out(wainee[6], papalaua_right[0], papalaua_left[1])

    w3 = Junction("w3")
    w3.set_roads_in(wainee[6], baker_right).set_roads_out(wainee[7])

    w4 = Junction("w4")
    w4.set_roads_in(wainee[7], kenui_left[1], kenui_right[2]).set_roads_out(kenui_right[1], kenui_left[2])


    ### LAHAINALUNA
    l1 = Junction("l1")
    l1.set_roads_in(luna_left[2], kuhua, luna_right[3]).set_roads_out(luna_right[2], luna_left[3])

    l2 = Junction("l2")
    l2.set_roads_in(luna_left[3], pauoa, luna_right[4]).set_roads_out(luna_right[3], luna_left[4])

    l3 = Junction("l3")
    l3.set_roads_in(luna_left[4], kale, luna_right[5]).set_roads_out(luna_right[4], luna_left[5])

    l4 = Junction("l4")
    l4.set_roads_in(luna_left[5], paunau_st, luna_right[6]).set_roads_out(luna_right[5], luna_left[6])

    l5_left = Junction("l5_left")
    l5_left.set_roads_in(luna_left[6]).set_roads_out(luna_left[7])

    l5_right = Junction("l5_right")
    l5_right.set_roads_in(luna_right[7]).set_roads_out(luna_right[6])

    l6 = Junction("l6")
    l6.set_roads_in(luna_left[7], kalena, luna_right[8]).set_roads_out(luna_right[7], luna_left[8])

    l7_left = Junction("l7_left")
    l7_left.set_roads_in(luna_left[8]).set_roads_out(luna_left[9])

    l7_right = Junction("l7_right")
    l7_right.set_roads_in(luna_right[9]).set_roads_out(luna_right[8])

    l8 = Junction("l8")
    l8.set_roads_in(luna_left[9], luna_right[10], bypass[0]).set_roads_out(luna_right[9], bypass[1])

    ### KEAWE
    k1 = Junction("k1")
    k1.set_roads_in(keawe[1]).set_roads_out(keawe[0])

    k2 = Junction("k2")
    k2.set_roads_in(keawe[2], komomai).set_roads_out(keawe[1])

    k3 = Junction("k3")
    k3.set_roads_in(bypass[1]).set_roads_out(keawe[2])

    ### KENUI - KAHOMA
    kenui_kahoma = Junction("kenui-kahoma")
    kenui_kahoma.set_roads_in(kenui_left[0], kenui_right[1], kahoma_village).set_roads_out(kenui_right[0], kenui_left[1])

    ### Creating network
    __network_am = Network([h1, h2, h3, h4, h5, f1, f2, f3, f4, f5, f6, w1, w2, w3, w4,
                    l1, l2, l3, l4, l5_left, l5_right, l6, l7_left, l7_right, l8,
                    k1, k2, k3, kenui_kahoma])

    __network_am.check_num_junctions(29)
    __network_am.check_roads_in_roadlist(road_list_am)

    network_am_base = copy.deepcopy(__network_am)
    road_list_am_base = copy.deepcopy(road_list_am)
    network_am_base.check_roads_in_roadlist(road_list_am_base)

    ## By default, h5 is the exit road, and we need to initialize the distances to it
    network_am_base.compute_distances(h5)  

    network_am_2 = copy.deepcopy(__network_am)
    road_list_am_2 = copy.deepcopy(road_list_am)

    to_remove = papalaua_left + [papalaua_right[1]]

    for r in to_remove:
        road_list_am_2.remove(r)
        network_am_2.remove_road(r)

    network_am_2.check_roads_in_roadlist(road_list_am_2)
    network_am_2.compute_distances(h5)  

    network_am_3 = copy.deepcopy(__network_am)
    road_list_am_3 = copy.deepcopy(road_list_am)

    to_remove = papalaua_left + [papalaua_right[1]] + luna_left[0:2] + [luna_right[1]]

    for r in to_remove:
        road_list_am_3.remove(r)
        network_am_3.remove_road(r)

    network_am_3.check_roads_in_roadlist(road_list_am_3)
    network_am_3.compute_distances(h5)  

    network_am_base.update_roads() 
    network_am_2.update_roads()
    network_am_3.update_roads()
    return network_am_base, network_am_2, network_am_3