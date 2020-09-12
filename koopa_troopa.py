###                KOOPA TROOPA                 
###                RST 9/11/2020                 
                                                
#  -------------------------------------------  
#  Required model files in directory:           
#  * action_prob.npy                            
#  * ev_params.csv                              
#  * mortality_means.pkl                        
#  * pattern2map.pkl                            
#  * pattern2_set.pkl                           
#  * ship_kill_means.pkl                        
#  * shipaction_model.txt                       
#  * shipyard_kill_means.pkl                    
#                                               
#  -------------------------------------------  
                                                
                                                
#  Common Code (from halite_turley.py)          
#  -------------------------------------------  
#                                               


                                     
# module with supporting functions for extracting features from a halite game
import os
import copy
import json
import time
import sys
import math
import random
import collections
import pickle

import pandas as pd
import numpy as np
import scipy.optimize
import scipy.ndimage
from scipy import signal

import lightgbm as lgb

# Global Variables
SHIP_ACTIONS = ['NORTH','SOUTH','EAST','WEST','CONVERT',None]


action_map = {None:0,
              'NORTH':1,
              'SOUTH':2,
              'EAST':3,
              'WEST':4,
              'CONVERT':5,
              'X':0}

action_map_reverse = {action_map[key]:key for key in action_map.keys()}

two_direction_map = {'XX':0,
                     'NN':1,
                     'NE':2, 
                     'NW':3,
                     'SS':4,
                     'SE':5,
                     'SW':6,
                     'EE':7,
                     'EN':8,
                     'ES':9,
                     'WW':10, 
                     'WN':11,
                     'WS':12, 
                     }

# DATA FILE VARIABLES
if 'FILE_DIR' not in locals():
    FILE_DIR = ''

with open(FILE_DIR+'pattern2_map.pkl', 'rb') as handle:
    pattern2_map = pickle.load(handle)
    
with open(FILE_DIR+'pattern2_set.pkl', 'rb') as handle:
    pattern2_set = pickle.load(handle)

def pattern2_index(local_patterns):
    # function that returns an array of pattern index values for a list of 'local_pattern_dist_2'
    return [pattern2_map[pattern_code]
            if pattern_code in pattern2_set
            else 32000
            for pattern_code in local_patterns]
    
action_prob = np.load(FILE_DIR+'action_prob.npy')

def pattern2_action_probabilities(local_patterns):
    return action_prob[pattern2_index(local_patterns)]


############################################################
mortality_means = pd.read_pickle(FILE_DIR+'mortality_means.pkl')
death_dict = mortality_means.to_dict()


def three_turn_mortality(pattern_list):
    #common_pattern = [pattern if pattern]
    return [death_dict[pattern] if pattern in death_dict else death_dict[-1] for pattern in pattern_list]

def exp_mortality(pattern_list, pattern_probabilities):
    return sum([death_dict[pattern_list[idx]]*pattern_probabilities[idx] 
                if pattern_list[idx] in death_dict
                else death_dict[-1]*pattern_probabilities[idx]
                for idx in range(len(pattern_list))])


############################################################
ship_kill_means = pd.read_pickle(FILE_DIR+'ship_kill_means.pkl')
ship_kill_dict = ship_kill_means.to_dict()

def exp_enemy_kills(pattern_list, pattern_probabilities):
    return sum([ship_kill_dict[pattern_list[idx]]*pattern_probabilities[idx] 
                if pattern_list[idx] in ship_kill_dict
                else ship_kill_dict[-1]*pattern_probabilities[idx]
                for idx in range(len(pattern_list))])


############################################################
shipyard_kill_means = pd.read_pickle(FILE_DIR+'shipyard_kill_means.pkl')
shipyard_kill_dict = shipyard_kill_means.to_dict()

def exp_shipyard_kill(pattern_list, pattern_probabilities):
    return sum([shipyard_kill_dict[pattern_list[idx]]*pattern_probabilities[idx] 
                if pattern_list[idx] in shipyard_kill_dict
                else shipyard_kill_dict[-1]*pattern_probabilities[idx]
                for idx in range(len(pattern_list))])


############################################################
# Expected Value Model
ev_params_df = pd.read_csv(FILE_DIR+'ev_params.csv',index_col=0)
ev_params = ev_params_df.to_numpy()


##############################################################
### Light GBM model
#action_forecast_model = lgb.Booster(model_file=FILE_DIR+'shipaction_model.txt')


# make a distance list that preserves the shortest travel route

def cardinal_distance(start_point,end_point,boardsize=21):
    # returns the distance needed to travel across a wrapped board of size [boardsize] where the 
    # first output is the west to east distance (or a negative value if faster to travel westbound)
    # and the second output is the north to south distance (or a negative value if shorter to 
    # travel southbound.
    #
    # The inputs, start_point and end_point are expected to be integers where value zero is the northwest
    # point on the board and value boardsize*boardsize-1 is the southeast point on the board.
    
    # Calculate the distance traveling east (1st element) or west (2nd element)
    dist_west2east = ((end_point - start_point) % boardsize, 
                      (boardsize - ( (end_point - start_point) % boardsize) ))
    # return the signed minimum distance, negative values means travel west
    dist_west2east = min(dist_west2east)*(-1)**dist_west2east.index(min(dist_west2east))

    # Calculate the distance traveling south (1st element) or north (2nd element)
    dist_north2south = ((end_point//boardsize - start_point//boardsize) % boardsize, 
                        ( boardsize - ( (end_point//boardsize - start_point//boardsize) % boardsize) ))
    # return the signed minimum distance, where negative values mean traveling north
    dist_north2south = min(dist_north2south)*(-1)**dist_north2south.index(min(dist_north2south))

    return dist_west2east, dist_north2south

def make_cardinal_distance_list(boardsize=21):
    startpoints = np.arange(boardsize**2)
    endpoints = np.arange(boardsize**2)
    cardinal_distance_list = []
    for start_point in startpoints:
        cardinal_distance_list.append([cardinal_distance(start_point,end_point) for end_point in endpoints])
    return cardinal_distance_list
cardinal_distance_list = make_cardinal_distance_list()

def make_total_distance_list(cardinal_distances=cardinal_distance_list, boardsize=21):
    startpoints = np.arange(boardsize**2)
    endpoints = np.arange(boardsize**2)
    total_distance_list = []
    for start_point in startpoints:
        total_distance_list.append([(abs(cardinal_distances[start_point][end_point][0]) + 
                                     abs(cardinal_distances[start_point][end_point][1]))
                                    for end_point in endpoints])
    return total_distance_list
total_distance_list = make_total_distance_list(cardinal_distance_list)

def destination_cell(start_point, move_distance = (0,0), boardsize=21):
    # returns the destination cell for a move distance tuple orderd in terms of
    # (move_west2east, move_north2south) so that a value of (1,3) moves on cell east and 3 cells
    # south and a value of (-3,-2) represents the cell that is 3 cells west and 2 cells to the north
    return ((start_point + move_distance[0]) % boardsize + 
            ((start_point//boardsize + move_distance[1])%boardsize) * boardsize)

def next_location(start_point,ship_action):
    # returns the destination cell for a ship that submits the ship_action
    if isinstance(ship_action,str):
        if ship_action.upper() == 'NORTH' or ship_action.upper() == 'N':
            next_cell = destination_cell(start_point, (0,-1))
        elif ship_action.upper() == 'SOUTH' or ship_action.upper() == 'S':
            next_cell = destination_cell(start_point, (0,1))
        elif ship_action.upper() == 'EAST' or ship_action.upper() == 'E':
            next_cell = destination_cell(start_point, (1,0))
        elif ship_action.upper() == 'WEST' or ship_action.upper() == 'W':
            next_cell = destination_cell(start_point, (-1,0))
        else:
            next_cell = start_point
    else:
        next_cell = start_point
    return next_cell


def next_directions(start_point, end_point):
    actions = []
    if start_point == end_point:
        actions.append(None)
    else:
        dist_west2east , dist_north2south = cardinal_distance_list[start_point][end_point]
        if dist_west2east>0:
            actions.append('EAST')
        elif dist_west2east<0:
            actions.append('WEST')
        if dist_north2south>0:
            actions.append('SOUTH')
        elif dist_north2south<0:
            actions.append('NORTH')
    return actions

def infer_ship_action(start_point,end_point, cardinal_distance_list=cardinal_distance_list):
    # returns the ship action that would take it from start_point to end_point
    cell_change = cardinal_distance_list[start_point][end_point]
    if cell_change[0] == 0 and cell_change[1] == -1:
        ship_action = 'NORTH'
    elif cell_change[0] == 0 and cell_change[1] == 1:
        ship_action = 'SOUTH'
    elif cell_change[0] == 1 and cell_change[1] == 0:
        ship_action = 'EAST'
    elif cell_change[0] == -1 and cell_change[1] == 0:
        ship_action = 'WEST'
    elif cell_change[0] == 0 and cell_change[1] == 0:
        ship_action = None
    else:
        raise Exception('move not possible')
    return ship_action
    
    
def cells_in_path(start_point, end_point, boardsize=21):
    # Output is a list of points that are on the shortest path between 
    # the start point and end point including the start and end points. 
    # All points are represented as integers, so the
    # point on the northwest corner has value 0 and the point two steps south
    # and two steps east has value 44. The function cells_in_path(0,66) outputs the 
    # nine points [0,1,2,21,22,23,42,43,44] on the path between them.
    west2east, north2south = cardinal_distance_list[start_point][end_point]
    if west2east<0:
        eaststep = -1
    else:
        eaststep = 1
    
    if north2south<0:
        southstep = -1
    else:
        southstep = 1
        
    startrowbase = boardsize*(start_point // boardsize)
    allrows = [rowval%boardsize for rowval in 
               range(start_point//boardsize,
                     (1 + start_point//boardsize + north2south),
                     southstep)]
    allcols = [colval%boardsize for colval in 
               range(start_point%boardsize,
                     (1 + start_point%boardsize + west2east),
                     eaststep)]
    path_cells = []
    for row in allrows:
        for col in allcols:
            path_cells.append(boardsize*row + col)
    return path_cells


def cells_in_distance(location, max_distance=1):
    # returns the index references for all cells within the specific distance
    celllist = [location]
    if max_distance>0:
        # for each distance from 1 to max_distance, find all cells
        for dist in range(1,1+max_distance):
            # rotate around all the combinations of west-east and north-wouth adjustments that equal distance=dist
            west2east = np.concatenate( (np.arange(0,dist,step=1),np.arange(dist,-dist,step=-1),np.arange(-dist,0,step=1)))
            north2south = np.concatenate( (np.arange(-dist,dist,step=1),np.arange(dist,-dist,step=-1)))
            for idx in range(len(west2east)):
                celllist.append(destination_cell(location,( west2east[idx],north2south[idx])))
    return celllist



def distance_kernel(maxdist):
    # Creates a matrix of ones where the distance from center is less than
    # or equal to maxdist 
    # e.g. distance_kernel(2) = [[0,1,0],[1,1,1],[0,1,0]]
    kernelmat = np.zeros((1+2*maxdist,1+2*maxdist))
    for i in range(1+2*maxdist):
        for j in range(1+2*maxdist):
            if abs(i-maxdist)+abs(j-maxdist)<=maxdist:
                kernelmat[i,j] = 1
    return kernelmat

def halite_sum_matrix(halite_data, maxdist = 0, boardsize = 21):
    # Creates a matrix from the stepdata representing the board
    # where each cell has the sum of all halite within maxdist.
    # The halite data is the raw format =(stepdata[0]['observation']['halite'])
    # halite_sum_matrix(stepdata, maxdist = 0, boardsize = 21)
    halite_matrix = np.reshape(halite_data,(boardsize,boardsize))
    return signal.convolve2d(halite_matrix, 
                             distance_kernel(maxdist), 
                             mode='same', boundary='wrap', fillvalue=0)

def halite_sum_array(halite_data, maxdist = 0, boardsize = 21):
    return np.reshape(halite_sum_matrix(halite_data, maxdist, boardsize),(boardsize*boardsize))

def halite_total_change(new_obs,old_obs):
    halite_changes = {}
    for team_num in range(4):
        for ship_id in list(new_obs['players'][team_num][2].keys()):
            if ship_id in list(old_obs['players'][team_num][2].keys()):
                halite_changes[ship_id] = (new_obs['players'][team_num][2][ship_id][1]
                                                     -old_obs['players'][team_num][2][ship_id][1] )
    return halite_changes


def last_mining_yield(prev_action,old_obs):
    halite_mined = {}
    for team_num in range(4):
        for ship_id in list(prev_action[team_num].keys()):
            if prev_action[team_num][ship_id] is None:
                ship_location = old_obs['players'][team_num][2][ship_id][0]
                halite_mined[ship_id] = int(0.25*old_obs['halite'][ship_location])
            else:
                halite_mined[ship_id] = 0
    
    return halite_mined


def present_prev_actions(ships_present,new_obs,old_obs):
    # infer each ship and shipyard action from the observation
    prev_action = [{},{},{},{}]

    for team_num in range(4):
        # Consider ships that still exist next turn
        for ship_id in list(new_obs['players'][team_num][2].keys()):
            if ship_id not in list(old_obs['players'][team_num][2].keys()):
                # 1) if didn't exist last turn, it was just spawned
                for shipyard_id in list(old_obs['players'][team_num][1].keys()):
                    if old_obs['players'][team_num][1][shipyard_id] == new_obs['players'][team_num][2][ship_id][0]:
                        prev_action[team_num][shipyard_id] = 'SPAWN'
            else:
                # 2) and if it did exist last turn, infer the action by its new location
                ship_move = infer_ship_action(start_point = old_obs['players'][team_num][2][ship_id][0],
                                  end_point = new_obs['players'][team_num][2][ship_id][0], 
                                  cardinal_distance_list=cardinal_distance_list)
                prev_action[team_num][ship_id] = ship_move
                        
    return prev_action

def investigate_mia(missing_ship_ids, new_obs, old_obs, verbose = False):
    # determines the last action and cause of death for ships that have gone missing
    
    # output variables
    missing_ship_last_action = {}
    missing_ship_killer_team = {}
    missing_ship_killer_uid = {}
    
    if len(missing_ship_ids)>0:
        
    
        # create list of missing ship team numbers, locations, and halite
        missing_ship_locations = {}
        missing_ship_halite = {}
        missing_ship_team_num = {}
        for ship_id in missing_ship_ids:
            for team_num in range(4):
                if ship_id in list(old_obs['players'][team_num][2].keys()):
                    missing_ship_team_num[ship_id] = team_num
                    missing_ship_locations[ship_id] = old_obs['players'][team_num][2][ship_id][0]
                    missing_ship_halite[ship_id] = old_obs['players'][team_num][2][ship_id][1]
        
        
        # infer the previous action for ships still present
        prev_action = present_prev_actions(list(new_obs['players'][team_num][2].keys()),new_obs,old_obs)
        
        # calculate the changes in halite for each ship
        halite_changes  = halite_total_change(new_obs,old_obs)
        halite_mined = last_mining_yield(prev_action,old_obs)
        for new_ship in new_obs['players'][team_num][2].keys():
            if new_ship not in halite_changes:
                halite_changes[new_ship] = 0
            if new_ship not in halite_mined:
                halite_mined[new_ship] = 0
        
        # deduce the last action, killer team and killer unit id for each missing ship among potential possibilities
        for ship_id in missing_ship_ids:
            potential_last_actions = []
            potential_killer_team = []
            potential_killer_uid = []

            adjacent_cells = [next_location(missing_ship_locations[ship_id],action_option) 
                              for action_option in ['N','S','E','W',None]]
            
            # for each adjacent cell, check if potential cause of death:
            #    - destroyed enemy shipyard if adjacent shipyard dissappeared (and it wasn't from another ship's attack)
            #    - was destroyed by an enemy ship if there is an enemy ship who now has its halite value
            #    - collided with enemy ship of equal halite value if adjacent enemy ship of same halite dissappeared
            #    - collided with friendly ship if ships of the same team dissappeared and were not killed by other means
            
            
            # converted to shipyard
            for team_num in range(4):
                for shipyard_id in list(new_obs['players'][team_num][1].keys()):
                    if shipyard_id not in list(old_obs['players'][team_num][1].keys()):
                        if (old_obs['players'][missing_ship_team_num[ship_id]][2][ship_id][0] 
                            == new_obs['players'][team_num][1][shipyard_id]):

                            potential_last_actions.append('CONVERT')
                            potential_killer_team.append(missing_ship_team_num[ship_id])
                            potential_killer_uid.append(ship_id)  # call it a suicide mission
                            # note if killer_uid is same as ship, it was convert to shipyard

            # **** Halite values visible in "obs" may not include the halite update from destroyed ships ****
            # destroyed by another ship who still exists and now has its halite value
            #for team_num in range(4):
                for other_ship in list(new_obs['players'][team_num][2].keys()):
                    try:
                        if ( (ship_id != other_ship) and                                                       # different ship 
                             (new_obs['players'][team_num][2][other_ship][0] in adjacent_cells) and           # in adjacent cell
                             (missing_ship_halite[ship_id] > 0) and                                                       # dead ship was not zero halite
                             ((halite_changes[other_ship] - halite_mined[other_ship]) == missing_ship_halite[ship_id])):  # halite increase = dead ship cargo

                            potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                            end_point=new_obs['players'][team_num][2][other_ship][0]))

                            potential_killer_team.append(team_num)
                            potential_killer_uid.append(other_ship)
                            # note this could be same team or different team, but killer_uid not the same as ship_id
                    except:
                        pass

            # destroyed by another ship who had same halite value and also died
            if len(missing_ship_ids)>1:
                for other_ship in missing_ship_ids:
                    if ( (other_ship is not ship_id) and
                         (missing_ship_halite[ship_id] == missing_ship_halite[other_ship]) and
                         (total_distance_list[missing_ship_locations[ship_id]][missing_ship_locations[other_ship]] < 3) ):
                            
                        other_adjacent_cells = [next_location(missing_ship_locations[other_ship],action_option) 
                              for action_option in ['N','S','E','W',None]]
                        collision_cells = [adjacent_cell for adjacent_cell in adjacent_cells if adjacent_cell in other_adjacent_cells]
                        ##### SMALL ERROR -- NO LOGIC TO DETERMINE WHICH COLLISION CELLS ARE VALID CHOICES, JUST CHOOSING FIRST OPTION
                        collision_cell = collision_cells[0]
                                                                                                          
                        potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                        end_point=collision_cell))
                        potential_killer_team.append(missing_ship_team_num[other_ship])
                        potential_killer_uid.append(other_ship)
                        # note this could be same team or different team, but killer_uid not the same as ship_id
                        
                        
            # destroyed enemy shipyard and was itself destroyed
            for team_num in range(4):
                if team_num != missing_ship_team_num[ship_id]:
                    for shipyard in list(old_obs['players'][team_num][1].keys()):
                        if (shipyard not in list(new_obs['players'][team_num][1].keys())) and (old_obs['players'][team_num][1][shipyard] in adjacent_cells):
                            potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                            end_point=old_obs['players'][team_num][1][shipyard]))
                            potential_killer_team.append(-1)      # -1 to signal shipyard kill
                            potential_killer_uid.append(shipyard)  # assign death to shipyard destroyed
                            # Note if team = -1, then shipyard destroyed

                            
                            
            # for the case where a ship is created in a shipyard and then immediately destroyed in battle
            for team_num in range(4):
                if team_num != missing_ship_team_num[ship_id]:
                    for shipyard_id in list(new_obs['players'][team_num][1].keys()):
                        if (total_distance_list[missing_ship_locations[ship_id]][new_obs['players'][team_num][1][shipyard_id]] == 1):
                            # assume that the ship attacked shipyard and was immediately killed by a newly spawned ship
                            potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                            end_point=new_obs['players'][team_num][1][shipyard_id]))
                            potential_killer_team.append(team_num)
                            potential_killer_uid.append(shipyard_id)  # we use the shipyard since we don't know the ship id

                            
            # destroyed by another ship with less halite 
            # potential error: change in in halite may not match ship cargo, and halite comparison lags step
            for team_num in range(4):
                for other_ship in list(new_obs['players'][team_num][2].keys()):

                    if ( (ship_id != other_ship) and                                                 # different ship
                         (new_obs['players'][team_num][2][other_ship][1] < old_obs['players'][missing_ship_team_num[ship_id]][2][ship_id][1]) and
                         (new_obs['players'][team_num][2][other_ship][0] in adjacent_cells)):       # in adjacent cell
                        
                        
                        potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                        end_point=new_obs['players'][team_num][2][other_ship][0] ))
                        potential_killer_team.append(team_num)
                        potential_killer_uid.append(other_ship)
                        # note this could be same team or different team, but killer_uid not the same as ship_id
                        
                        
            
            # for the case where a ship converts to a shipyard before being killed
            if len(missing_ship_ids)>1:
                for other_ship in missing_ship_ids:
                    if ( (other_ship is not ship_id) and
                         (total_distance_list[missing_ship_locations[ship_id]][missing_ship_locations[other_ship]] == 1) ):
                        
                        if missing_ship_halite[ship_id] > missing_ship_halite[other_ship]: # missing_ship converted
                            collision_cell = old_obs['players'][missing_ship_team_num[ship_id]][2][ship_id][0]
                            potential_last_actions.append('CONVERT')
                            potential_killer_team.append(missing_ship_team_num[ship_id])
                            potential_killer_uid.append(ship_id)
                        else:  # other ship converted
                            collision_cell = old_obs['players'][missing_ship_team_num[other_ship]][2][other_ship][0]
                            potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                            end_point=collision_cell))
                            potential_killer_team.append(missing_ship_team_num[other_ship]-10)
                            potential_killer_uid.append(missing_ship_team_num[other_ship])
                            
            # do a version that does not check halite to account for halite counting errors in mining or concurrent battles
            for team_num in range(4):
                for other_ship in list(new_obs['players'][team_num][2].keys()):
                    if ( (ship_id != other_ship) and                                                 # different ship
                         (new_obs['players'][team_num][2][other_ship][0] in adjacent_cells)):       # in adjacent cell
                                                      
                        potential_last_actions.append(infer_ship_action(start_point=missing_ship_locations[ship_id],
                                                                        end_point=new_obs['players'][team_num][2][other_ship][0] ))
                        potential_killer_team.append(team_num)
                        potential_killer_uid.append(other_ship)
                        # note this could be same team or different team, but killer_uid not the same as ship_id
                        
            
            if len(potential_last_actions)>0:
                # When there are multiple potential options, select the first (highest priority)
                missing_ship_last_action[ship_id] = potential_last_actions[0]
                missing_ship_killer_team[ship_id] = potential_killer_team[0]
                missing_ship_killer_uid[ship_id] = potential_killer_uid[0]
            else:
                # cases where no cause of death can be discerned (e.g. player timeout and forfeit)
                missing_ship_last_action[ship_id] = None
                missing_ship_killer_team[ship_id] = missing_ship_team_num[ship_id]
                missing_ship_killer_uid[ship_id] = ship_id
                
            
    if verbose:
        if len(missing_ship_ids)>0:
            for ship_id in missing_ship_ids:
 
                if missing_ship_last_action[ship_id] is None:
                    print_action = 'MINE'
                else:
                    print_action = missing_ship_last_action[ship_id]
                print(ship_id + ' at cell ' + str(missing_ship_locations[ship_id]) + 
                      ' chose action ' + print_action + 
                      ' to cell ' + str(next_location(missing_ship_locations[ship_id],missing_ship_last_action[ship_id])) +
                      ' and was killed by ' + missing_ship_killer_uid[ship_id] +
                      ' from team ' + str(missing_ship_killer_team[ship_id]) )
    
    return missing_ship_last_action, missing_ship_killer_team, missing_ship_killer_uid                           


def infer_previous_action(new_obs, old_obs, verbose = False):
    # infer previous actions from the observations
    
    present_ship_ids = []
    missing_ship_ids = []
    for team_num in range(4):
        present_ship_ids += list(new_obs['players'][team_num][2].keys())
        missing_ship_ids += [ship_id for ship_id in list(old_obs['players'][team_num][2].keys()) if ship_id not in present_ship_ids]
    
    prev_actions = present_prev_actions(present_ship_ids,new_obs,old_obs)
    missing_ship_last_action, _, _ = investigate_mia(missing_ship_ids, new_obs, old_obs, verbose = verbose)
    
    for team_num in range(4):
        prev_actions[team_num].update({missing_ship_id:missing_ship_last_action[missing_ship_id] 
                                      for missing_ship_id in list(old_obs['players'][team_num][2].keys()) 
                                       if missing_ship_id not in list(new_obs['players'][team_num][2].keys())}) 
    return prev_actions


def nearby_features(ship_id, team, obs, max_distance=3):
    
    # outputs the pattern value for the distance around the ship
    ship_location = obs['players'][team][2][ship_id][0]
    ship_halite = obs['players'][team][2][ship_id][1]
    nearby_cells = cells_in_distance(ship_location,max_distance)
    nearby_features_array = np.zeros(shape=len(nearby_cells),dtype=np.uint64)
    
    # 1 - Friendly shipyard
    for shipyard_location in list(obs['players'][team][1].values()):
        if shipyard_location in nearby_cells:
            nearby_features_array[nearby_cells.index(shipyard_location)] = 1
        
    # 2 - Friendly ships
    friendly_ship_locations = [shipdata[0] for shipdata in list(obs['players'][team][2].values())]
    for friendly_ship_location in friendly_ship_locations:
        if friendly_ship_location in nearby_cells and friendly_ship_location is not ship_location:
            nearby_features_array[nearby_cells.index(friendly_ship_location)] = 2
    
    
    for other_team in range(4):
        if other_team is not team:
    
            # 3 - Enemy shipyards
            for shipyard_location in list(obs['players'][other_team][1].values()):
                if shipyard_location in nearby_cells:
                    nearby_features_array[nearby_cells.index(shipyard_location)] = 3
                    
            # 4 - Enemy fat ships
            enemy_fleet_data = list(obs['players'][other_team][2].values())
            for enemy_ship_data in enemy_fleet_data:
                if (enemy_ship_data[0] in nearby_cells) and (enemy_ship_data[1] > ship_halite):
                    nearby_features_array[nearby_cells.index(enemy_ship_data[0])] = 4
                    
            # 5 - Enemy lean ships
            for enemy_ship_data in enemy_fleet_data:
                if (enemy_ship_data[0] in nearby_cells) and (enemy_ship_data[1] <= ship_halite):
                    nearby_features_array[nearby_cells.index(enemy_ship_data[0])] = 5
                    
    return nearby_features_array


def rotational_equivalents3d(nearby_features_array):
    # returns indices for "nearby_features" indices that are strategically equivalent, 
    # simply rotated or reflected game boards
    # order of manipulations is: rotation counter-clockwise once, twice, three-times, 
    # mirror left/right, mirror up/down, mirror nortwest/southest, mirror northeast/southwest
    
    rotations = np.array([[0,0,0,0,0,0,0,0],
                            [1,2,3,4,1,3,2,4],
                            [2,3,4,1,4,2,1,3],
                            [3,4,1,2,3,1,4,2],
                            [4,1,2,3,2,4,3,1],
                            [5,7,9,11,5,9,7,11],
                            [6,8,10,12,12,8,6,10],
                            [7,9,11,5,11,7,5,9],
                            [8,10,12,6,10,6,12,8],
                            [9,11,5,7,9,5,11,7],
                            [10,12,6,8,8,12,10,6],
                            [11,5,7,9,7,11,9,5],
                            [12,6,8,10,6,10,8,12],
                            [13,16,19,22,13,19,16,22],
                            [14,17,20,23,24,18,15,21],
                            [15,18,21,24,23,17,14,20],
                            [16,19,22,13,22,16,13,19],
                            [17,20,23,14,21,15,24,18],
                            [18,21,24,15,20,14,23,17],
                            [19,22,13,16,19,13,22,16],
                            [20,23,14,17,18,24,21,15],
                            [21,24,15,18,17,23,20,14],
                            [22,13,16,19,16,22,19,13],
                            [23,14,17,20,15,21,18,24],
                            [24,15,18,21,14,20,17,23]])
    
    rotational_indices = []
    for r in range(8):
        rotational_indices.append( [nearby_features_array[rotations[idx,r]] for idx in range(25)] )
    
    return rotational_indices

def rotational_equivalents2d(nearby_features_array):
    # returns indices for "nearby_features" indices that are strategically equivalent, 
    # simply rotated or reflected game boards
    # order of manipulations is: rotation counter-clockwise once, twice, three-times, 
    # mirror left/right, mirror up/down, mirror nortwest/southest, mirror northeast/southwest
    
    rotations = np.array([[0,0,0,0,0,0,0,0],
                            [1,2,3,4,1,3,2,4],
                            [2,3,4,1,4,2,1,3],
                            [3,4,1,2,3,1,4,2],
                            [4,1,2,3,2,4,3,1],
                            [5,7,9,11,5,9,7,11],
                            [6,8,10,12,12,8,6,10],
                            [7,9,11,5,11,7,5,9],
                            [8,10,12,6,10,6,12,8],
                            [9,11,5,7,9,5,11,7],
                            [10,12,6,8,8,12,10,6],
                            [11,5,7,9,7,11,9,5],
                            [12,6,8,10,6,10,8,12]])
    
    rotational_indices = []
    for r in range(8):
        rotational_indices.append( [nearby_features_array[rotations[idx,r]] for idx in range(25)] )
    
    return rotational_indices

# transform the nearby cells into a single integer
def nearby_feature_code3d(nearby_features_array):
    return np.sum( np.multiply(nearby_features_array[1:],     # skip first element which contains the ship itself
                               np.power(6, np.arange(0,24,dtype=np.uint64))))



# transform the nearby cells into a single integer
def nearby_feature_code2d(nearby_features_array):
    return np.sum( np.multiply(nearby_features_array[1:],     # skip first element which contains the ship itself
                               np.power(6, np.arange(0,12,dtype=np.uint64))))




def nearest_shipyard(start_point, team_num, obs):
    closest_shipyard_position = None
    closest_shipyard_distance = 999
    for shipyard_id in obs['players'][team_num][1]:
        if total_distance_list[start_point][obs['players'][team_num][1][shipyard_id]] < closest_shipyard_distance:
            closest_shipyard_position = obs['players'][team_num][1][shipyard_id]
            closest_shipyard_distance = total_distance_list[start_point][obs['players'][team_num][1][shipyard_id]]
    return closest_shipyard_distance, closest_shipyard_position


def remove_none(turn_dict):
    return {key:turn_dict[key] for key in turn_dict if turn_dict[key] != None}

#########################################################################################################
### OPTIMUS (https://www.kaggle.com/solverworld/optimus-mine-agent)

#See notebook on optimal mining kaggle.com/solverworld/optimal-mining
turns_optimal = np.array(
                          [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
                           [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
                           [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
                           [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
                           [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
                           [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def limit(x,a,b):
  if x<a:
    return a
  if x>b:
    return b
  return x

def num_turns_to_mine(C,H,rt_travel):
    #How many turns should we plan on mining?
    #C=carried halite, H=halite in square, rt_travel=steps to square and back to shipyard
    if C==0:
        ch=0
    elif H==0:
        ch=turns_optimal.shape[0]
    else:
        ch=int(math.log(C/H)*2.5+5.5)
        ch=limit(ch,0,turns_optimal.shape[0]-1)
    rt_travel=int(limit(rt_travel,0,turns_optimal.shape[1]-1))
    return turns_optimal[ch,rt_travel]

def halite_per_turn(carrying, halite,travel,min_mine=1):
    #compute return from going to a cell containing halite, using optimal number of mining steps
    #returns halite mined per turn, optimal number of mining steps
    #Turns could be zero, meaning it should just return to a shipyard (subject to min_mine)
    turns=num_turns_to_mine(carrying,halite,travel)
    if turns<min_mine:
        turns=min_mine
    mined=carrying+(1-.75**turns)*halite
    return mined/(travel+turns), turns

def optimus_objectives(team_num, obs):
    #Assign the ships to a cell containing halite optimally
    #set ship_objectives[ship_id] to a Position
    #We assume that we mine halite containing cells optimally or return to deposit
    #directly if that is optimal, based on maximizing halite per step.
    #Make a list of pts containing cells we care about, this will be our columns of matrix C
    #the rows are for each ship in collect
    #computes global dict ship_tagert with shipid->Position for its target
    #global ship targets should already exist
    
    
    halite_min = 50
    num_shipyard_targets = 4
    
    ship_objectives={}
    
    # pts1 is every halite deposite with greater than halite_min halite
    pts1=[idx for idx in range(21*21) if obs['halite'][idx]>halite_min]
    # pts2 is the location of every shipyard
    pts2=[obs['players'][team_num][1][sy] for sy in obs['players'][team_num][1]]*num_shipyard_targets
    pts=pts1+pts2
    
    #this will be the value of assigning C[ship,pt]
    #ship_ids = [ship.ship_id for ship in STATE.ships]
    C=np.zeros((len(obs['players'][team_num][2]),len(pts1)+len(pts2)))
    #this will be the optimal mining steps we calculated
    ship_ids = list(obs['players'][team_num][2].keys())
    for i,ship_id in enumerate(obs['players'][team_num][2]):
        for j,pt in enumerate(pts):
            #two distances: from ship to halite, from halite to nearest shipyard
            d1 = total_distance_list[obs['players'][team_num][2][ship_id][0]][pt]
            d2, shipyard_position = nearest_shipyard(pt, team_num = team_num, obs=obs)

            #value of target is based on the amount of halite per turn we can do
            if j < len(pts1):  #if calculating value of mining
                v, mining=halite_per_turn(obs['players'][team_num][2][ship_id][1],
                                          obs['halite'][pt], d1+d2)
            else: # calculating the value of traveling to shipyard
                if d1>0:
                  v= obs['players'][team_num][2][ship_id][1] / d1
                else:
                  v=0
            # The existing logic around an enemy being on the target square seems awkward!
            enemy_ship_halite = {}
            for team in range(4):
                if team != team_num:
                    for enemy_ship in list(obs['players'][team][2].keys()):
                        enemy_ship_halite[obs['players'][team][2][enemy_ship][0]] = obs['players'][team][2][enemy_ship][1]
                        
            if pt in list(enemy_ship_halite.keys()):
                #if someone else on the cell, see how much halite they have
                if enemy_ship_halite[pt] <= obs['players'][team_num][2][ship_id][1]:
                  v = -1000   # don't want to go there
                else:
                  if d1<3:
                    #attack or scare off if reasonably quick to get there
                    v+= enemy_ship_halite[pt] / (d1+1)  # want to attack them or scare them off
            C[i,j]=v
            
    row,col=scipy.optimize.linear_sum_assignment(C, maximize=True)
    #so ship row[i] is assigned to target col[j]
    
    for r,c in zip(row,col):
        ship_objectives[ship_ids[r]]=pts[c]
  
    return ship_objectives
###################################################################################################################################


    
class CumulativeData:
    def __init__(self, new_obs):
        self.step = 0
        self.obs = new_obs
        self.old_obs = new_obs
        self.all_ship_names = [] # list of all past and preesnt ship names in game, index of this list is shipNumId
        self.all_ship_numbers = [] # list of all past and present ship names in game (aka shipNumId)
        self.all_ship_team_num = {} # dictionary of ship_names as keys and team numbers as values
        for team_num in range(4):
            for ship_name in list(new_obs['players'][team_num][2].keys()):
                self.all_ship_names.append(ship_name)
                self.all_ship_numbers.append(self.all_ship_names.index(ship_name))
                self.all_ship_team_num[ship_name] = team_num
        
        self.team_ship_converts = [0,0,0,0]   # ship converts to shipyard (gone)
        self.team_ship_collides = [0,0,0,0]   # ship collides with friendly (gone)
        self.team_ship_deaths = [0,0,0,0]     # ship destroyed by enemy (gone)
        self.team_shipyard_births = [0,0,0,0]     # ship destroyed by enemy (gone)
        self.team_shipyard_kills = [0,0,0,0]  # ship destroys enemy shipyard (gone)
        self.team_ship_births = [0,0,0,0]     # new ship created by friendly shipyard
        self.team_ship_kills = [0,0,0,0]      # enemy ship destroyed by team
        self.team_shipyard_deaths = [0,0,0,0] # friendly shipyard destroyed by enemy
        self.team_shipyard_saves = [0,0,0,0]  # saved a friendly shipyard from death
        
        self.team_actions_previous = [{},{},{},{}]
        self.team_actions_2steps_ago = [{},{},{},{}]
        self.team_actions_3steps_ago = [{},{},{},{}]
        
        
        
    def update(self, new_obs):
        for team_num in range(4):
            for ship_name in list(new_obs['players'][team_num][2].keys()):
                if ship_name not in self.all_ship_names:
                    self.all_ship_names.append(ship_name)
                    self.all_ship_numbers.append(self.all_ship_names.index(ship_name))
                    self.all_ship_team_num[ship_name] = team_num
                    
        team_actions = infer_previous_action(new_obs, self.obs, verbose = False)
        self.team_actions_3steps_ago = self.team_actions_2steps_ago
        self.team_actions_2steps_ago = self.team_actions_previous
        self.team_actions_previous = team_actions
        
        mia_ships = []
        for team_num in range(4):
            self.team_ship_births[team_num] += sum([ship_act == 'SPAWN' 
                                                    for ship_act in team_actions[team_num].values()])
            
            self.team_shipyard_births[team_num] += len([shipyard for shipyard 
                                                        in list(new_obs['players'][team_num][1].keys()) 
                                                        if shipyard not in list(self.obs['players'][team_num][1].keys())])
            
            self.team_shipyard_deaths[team_num] += len([shipyard for shipyard 
                                                        in list(self.obs['players'][team_num][1].keys()) 
                                                        if shipyard not in list(new_obs['players'][team_num][1].keys())])
            
            # list of missing ships
            mia_ships += [ship_id for ship_id in self.obs['players'][team_num][2] if ship_id not in new_obs['players'][team_num][2]]
            
        # determine cause of death and tally it up
        if len(mia_ships)>0:
            ms_action, ms_killer_team, ms_killer_uid  = investigate_mia(mia_ships, new_obs, self.obs, verbose = False)
            for ship_id in mia_ships:
                if ms_killer_team[ship_id] < 0:                 # killed a shipyard
                    self.team_shipyard_kills[self.all_ship_team_num[ship_id]] += 1
                    if ms_killer_team[ship_id] < -1:   # killed shipyard that was simultaneously created
                        self.team_ship_converts[ms_killer_team[ship_id]+10] += 1
                        self.team_shipyard_births[ms_killer_team[ship_id]+10] += 1
                        self.team_shipyard_deaths[ms_killer_team[ship_id]+10] += 1
                elif ms_killer_team[ship_id] == self.all_ship_team_num[ship_id]:         # killed by same team...
                    if ms_killer_uid[ship_id] == ship_id:
                        self.team_ship_converts[self.all_ship_team_num[ship_id]] += 1    # as a result of converting to shipyard
                    else:
                        self.team_ship_collides[self.all_ship_team_num[ship_id]] += 1         # as a result of a same-team collision
                else:
                    self.team_ship_deaths[self.all_ship_team_num[ship_id]] += 1
                    self.team_ship_kills[ms_killer_team[ship_id]] += 1
                    if ms_killer_uid[ship_id] in self.obs['players'][ms_killer_team[ship_id]][2]:
                        killer_location = next_location(start_point = self.obs['players'][ms_killer_team[ship_id]][2][ms_killer_uid[ship_id]][0],
                                                    ship_action = ms_action[ship_id])
                        # tally if the killer saved a shipyard
                        if (killer_location in list(new_obs['players'][ms_killer_team[ship_id]][1].values())):    
                            self.team_shipyard_saves[ms_killer_team[ship_id]] += 1 
                            
        self.old_obs = self.obs
        self.obs = new_obs
    
    def __repr__(self):
        outstring = 'class:CumulativeData \n'
        if len(self.all_ship_names)>12:
            for key,value in self.__dict__.items():
                if key == 'obs':
                    outstring += key + ': Halite game observation data for step ' +str(self.step) + ' \n'
                elif key == 'old_obs':
                    outstring += key + ': Halite game observation data for step ' +str(self.old_obs['step']) + ' \n'
                elif key == 'all_ship_names':
                    outstring += key + ': ' +str(len(value)) + ' ship uid values \n'
                elif key == 'all_ship_numbers':
                    outstring += key + ': ' +str(len(value)) + ' unique integer identifiers \n'
                elif key == 'all_ship_team_num':
                    outstring += key + ': dictionary of ' +str(len(value)) + ' ship names mapped to team numbers \n'
                elif key == 'team_actions_previous':
                    outstring += key + ': 4-element list of dictionaries of team ship names mapped to ship actions \n'
                elif key == 'team_actions_2steps_ago':
                    outstring += key + ': 4-element list of dictionaries of team ship names mapped to ship actions \n'
                elif key == 'team_actions_3steps_ago':
                    outstring += key + ': 4-element list of dictionaries of team ship names mapped to ship actions \n'  
                    
                else:
                    outstring += key + ':' + str(value) + '\n'
        else:
            for key,value in self.__dict__.items():
                if key == 'obs':
                    outstring += key + ': Halite game observation data for step ' +str(self.obs['step']) + ' \n'
                elif key == 'old_obs':
                    outstring += key + ': Halite game observation data for step ' +str(self.old_obs['step']) + ' \n'
                else:
                    outstring += key + ':' + str(value) + '\n'
        return outstring
    
    

class HaliteBoard:
    def __init__(self, new_obs):
        self.board_size = 21
        self.total_halite = int(sum(new_obs['halite']))
        self.total_ships = sum([len(new_obs['players'][team_num][2]) for team_num in range(4)])
        self.total_shipyards = sum([len(new_obs['players'][team_num][1]) for team_num in range(4)])
        
        self.halite_map = new_obs['halite']
        
        self.friend_present_map = [0]*(21**2)
        for ship in list(new_obs['players'][new_obs['player']][2].values()):
            self.friend_present_map[ship[0]] = 1
        
        self.enemy_present_map = [0]*(21**2)
        for team_num in range(4):
            if team_num != new_obs['player']:
                for ship in list(new_obs['players'][team_num][2].values()):
                    self.enemy_present_map[ship[0]] = 1
                
        #self.ship_map = [-1 for cellnum in range(self.board_size**2)]
        #self.shipyard_map = [-1 for cellnum in range(self.board_size**2)]
        #for team_num in range(4):
        #    for shipyard_data in list(new_obs['players'][team_num][1].values()):
        #        self.shipyard_map[shipyard_data] = team_num
        #    for ship_data in list(new_obs['players'][team_num][2].values()):
        #        self.ship_map[ship_data[0]] = team_num
        
    def __repr__(self):
        outstring = 'class:BOARD \n'
        for key,value in self.__dict__.items():
            if key == 'halite_at_distance':
                outstring += key + ':' + str(len(value)) + ' arrays \n'
            elif key == 'halite_map':
                outstring += '  ' + key + ': array with locations of ' + str(self.total_halite) + ' halite on map \n'    
            elif key == 'ship_map':
                outstring += '  ' + key + ': array with locations of ' + str(self.total_ships) + ' ships on map \n'
            elif key == 'enemy_present_map':
                outstring += '  ' + key + ': array where 1 means enemy ship in location \n'    
            elif key == 'shipyard_map':
                outstring += '  ' + key + ': array with locations of ' + str(self.total_shipyards) + ' shipyards on map \n'
            else:
                outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring


class Ship:
    def __init__(self, ship_id = '', location=None, halite=0, team=None, probability=0.0):
        self.ship_id = ship_id
        self.location = location
        self.halite = halite
        self.team = team
        self.probability = probability
        
    def __repr__(self):
        outstring = 'class:Ship \n'
        for key,value in self.__dict__.items():
            outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring
    

class Shipyard:
    def __init__(self, shipyard_id = '', location=None, team=None, probability=0.0):
        self.shipyard_id = shipyard_id
        self.ship_id = shipyard_id
        self.location = location
        self.team = team
        self.probability = probability
        
    def __repr__(self):
        outstring = 'class:Shipyard \n'
        for key,value in self.__dict__.items():
            outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring
    

            
            
class Turn:
    def __init__(self, new_obs):
        self.start_time = time.time()
        self.next_actions = {}
        self.next_ship_locations = []
        self.shipyard_locations = new_obs['players'][new_obs['player']][1]
        self.team_ships = len(new_obs['players'][new_obs['player']][2])
        self.team_halite = new_obs['players'][new_obs['player']][0]
    
    def __repr__(self):
        outstring = 'class:Turn \n'
        for key,value in self.__dict__.items():
            outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring
    
def ship_neg_halite(ship):
    # used for sorting purposes
    return ship.halite*-1
    
class State:
    def __init__(self, new_obs):
        self.obs = new_obs
        self.prev_obs = new_obs
        self.board = HaliteBoard(new_obs)
        self.team = new_obs['player']
        self.ships = []
        self.shipyards = []
        self.enemy_ships = []
        self.enemy_shipyards = []
        self.cumulative_data = CumulativeData(new_obs)
        self.ship_data =  []
        
        
    def update(self, new_obs):
        self.prev_obs = self.obs.copy()
        self.obs = new_obs
        self.board = HaliteBoard(new_obs)
        self.ships = []
        self.shipyards = []
        self.enemy_ships = []
        self.enemy_shipyards = []
        self.cumulative_data.update(new_obs=new_obs)
        self.ship_data =  pd.DataFrame(create_ship_game_features(new_obs = new_obs, cumulative_data = self.cumulative_data))
        
        for team_num in range(4):
            if team_num == new_obs['player']:
                for shipyard in list(new_obs['players'][team_num][1].keys()):
                    self.shipyards.append(Shipyard(shipyard_id=shipyard,
                                                   team = team_num,
                                                   location=new_obs['players'][team_num][1][shipyard],
                                                   probability = 1.0))
                    
                for ship in new_obs['players'][team_num][2]:
                    self.ships.append(Ship(ship_id=ship,
                                           location=new_obs['players'][team_num][2][ship][0],
                                           halite = new_obs['players'][team_num][2][ship][1],
                                           team = team_num,
                                           probability = 1.0))
                    
            else:
                for shipyard in list(new_obs['players'][team_num][1].keys()):
                    self.enemy_shipyards.append(Shipyard(shipyard_id=shipyard,
                                                         team = team_num,
                                                         location=new_obs['players'][team_num][1][shipyard],
                                                         probability = 1.0))
                    
                for ship in new_obs['players'][team_num][2]:
                    self.enemy_ships.append(Ship(ship_id=ship,
                                                 location=new_obs['players'][team_num][2][ship][0],
                                                 halite = new_obs['players'][team_num][2][ship][1],
                                                 team = team_num,
                                                 probability = 1.0))
                    
        # sort list of ships so that the highest value ship goes first
        self.ships.sort(key=ship_neg_halite)
        
        
                    
        
    def __repr__(self):
        outstring = 'class:State \n'
        for key,value in self.__dict__.items():
            if key == 'obs':
                outstring += '  ' + key + ': Halite "obs" for step ' + str(self.obs['step']) + ' \n'
            elif key == 'prev_obs':
                outstring += '  ' + key + ': Halite "obs" for step ' + str(self.prev_obs['step']) + ' \n' 
            elif key == 'board':
                outstring += '  ' + key + ': HaliteBoard object \n'
            elif key == 'ships':
                outstring += '  ' + key + ': ' + str(len(self.ships)) + ' Ship objects \n'
            elif key == 'enemy_ships':
                outstring += '  ' + key + ': ' + str(len(self.enemy_ships)) + ' Ship objects \n'
            elif key == 'shipyards':
                outstring += '  ' + key + ': ' + str(len(self.shipyards)) + ' Shipyard objects \n'
            elif key == 'enemy_shipyards':
                outstring += '  ' + key + ': ' + str(len(self.enemy_shipyards)) + ' Shipyard objects \n'
            else:
                outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring

    
class BoardOutputGame:
    def __init__(self, new_obs):
        self.board_size = 21
        self.board_halite_total = int(sum(new_obs['halite']))
        self.board_ship_total = sum([len(new_obs['players'][team_num][2]) for team_num in range(4)])
        
    def __repr__(self):
        outstring = 'class:BoardOutputGame \n'
        for key,value in self.__dict__.items():
            if key == 'board_halite_at_distance':
                outstring += key + ':' + str(len(value)) + ' arrays \n'
            elif key == 'board_ship_map':
                outstring += key + ':' + str(self.board_ship_total) + ' ships on map \n'
            elif key == 'board_shipyard_map':
                outstring += key + ':' + str(self.board_shipyard_total) + ' shipyards on map \n'
            else:
                outstring += key + ':' + str(value) + '\n'
        return outstring
    
    

class TeamOutputGame:
    def __init__(self, new_obs):
        self.team_list = [0,1,2,3]
        self.team_halite = [new_obs['players'][team_num][0] for team_num in range(4)]
        self.team_cargo = [sum([shipdata[1] for shipdata in new_obs['players'][team_num][2].values()]) for team_num in range(4)]
        self.team_shipyards = [len(new_obs['players'][team_num][1]) for team_num in range(4)]
        self.team_ships = [len(new_obs['players'][team_num][2]) for team_num in range(4)]
        
    def __repr__(self):
        outstring = 'class:TeamOutputGame \n'
        for key,value in self.__dict__.items():
            outstring += key + ':' + str(value) + '\n'
        return outstring
    
    
    
class ShipOutputGame:
    def __init__(self, new_obs, include_future = False):
        
        ### Initialize Variables
        self.ship_list = []
        self.ship_location = []

        self.ship_team_num = []
 
        self.ship_cargo = []
        self.ship_dist_shipyard = []
        self.ship_dist_enemy_yard = []
        self.ship_direction_shipyard = []
        
        self.ship_halite_d0 = []
        self.ship_halite_d1 = []
        self.ship_halite_d2 = []
        self.ship_halite_d3 = []
        
        self.ship_halite_north = []
        self.ship_halite_south = []
        self.ship_halite_east = []
        self.ship_halite_west = []
        
        self.ship_friendly_d1 = []
        self.ship_friendly_d2 = []
        self.ship_friendly_d3 = []
        self.ship_friendly_d4 = []

        self.ship_fat_enemy_d1 = []
        self.ship_fat_enemy_d2 = []
        self.ship_fat_enemy_d3 = []
        self.ship_fat_enemy_d4 = []

        self.ship_lean_enemy_d1 = []
        self.ship_lean_enemy_d2 = []
        self.ship_lean_enemy_d3 = []
        self.ship_lean_enemy_d4 = []
        self.ship_lean_blocking = []


        self.ship_pattern2d = []
    
        ### Calculate Values
        
        # Calculate sum of halite for various distances relative to each board cell
        board_halite_at_distance = []
        for d in range(4):
            board_halite_at_distance.append( halite_sum_array(new_obs['halite'], maxdist = d) )
        
        # Pull data for each team, calculating locations of friendly and enemy units
        for team_num,team_data in enumerate(new_obs['players']):

            # make a list of friendly and enemy shipyard locations
            friendly_shipyard_locations = list(team_data[1].values())

            enemy_shipyard_locations = []
            for enemy_num in range(4):
                if enemy_num is not team_num:
                    enemy_shipyard_locations += list(new_obs['players'][enemy_num][1].values())

            # make lists of friendly and enemy ships
            friendly_ship_locations = [shipdata[0] for shipdata in list(team_data[2].values())]
            friendly_ship_array = [1 if cellidx in friendly_ship_locations else 0 for cellidx in range(21**2)]

            enemy_ship_locations = []
            enemy_ship_cargo = []
            enemy_ship_data = []
            for enemy_num in range(4):
                if enemy_num is not team_num:
                    for shipdata in list(new_obs['players'][enemy_num][2].values()):
                        enemy_ship_data.append(shipdata)
                        enemy_ship_locations.append(shipdata[0])
                        enemy_ship_cargo.append(shipdata[1])


            # SHIP-LEVEL DATA
            for ship_name in list(team_data[2].keys()):
                self.ship_list.append(ship_name)
                self.ship_team_num.append(team_num)               
                self.ship_cargo.append(team_data[2][ship_name][1])
                
                
                # Calculate distances to units
                ship_location = team_data[2][ship_name][0]
                self.ship_location.append(ship_location)
                
                shipyard_direction = 'XX'
                if len(friendly_shipyard_locations)<1:
                    friendly_shipyard_distances = [21]
                    closest_shipyard = None
                else:
                    friendly_shipyard_distances = [total_distance_list[ship_location][shipyards] 
                                                   for shipyards in friendly_shipyard_locations]
                    closest_shipyard = friendly_shipyard_locations[friendly_shipyard_distances.index(min(friendly_shipyard_distances))]
                    cardinal_distances = cardinal_distance_list[ship_location][closest_shipyard]
                    if cardinal_distances[0] == 0:
                        if cardinal_distances[1]>0:
                            shipyard_direction = 'SS'
                        elif cardinal_distances[1]<0:
                            shipyard_direction = 'NN'
                    elif cardinal_distances[1] == 0:
                        if cardinal_distances[0]>0:
                            shipyard_direction = 'EE'
                        elif cardinal_distances[0]<0:
                            shipyard_direction = 'WW'
                    else:
                        if abs(cardinal_distances[0]) >= abs(cardinal_distances[1]):
                            if (cardinal_distances[0]>0) and (cardinal_distances[1]>0):
                                shipyard_direction = 'ES'
                            if (cardinal_distances[0]>0) and (cardinal_distances[1]<0):
                                shipyard_direction = 'EN'
                            if (cardinal_distances[0]<0) and (cardinal_distances[1]>0):
                                shipyard_direction = 'WS'
                            if (cardinal_distances[0]<0) and (cardinal_distances[1]<0):
                                shipyard_direction = 'WN'
                        elif abs(cardinal_distances[0]) < abs(cardinal_distances[1]):
                            if (cardinal_distances[0]>0) and (cardinal_distances[1]>0):
                                shipyard_direction = 'SE'
                            if (cardinal_distances[0]>0) and (cardinal_distances[1]<0):
                                shipyard_direction = 'NE'
                            if (cardinal_distances[0]<0) and (cardinal_distances[1]>0):
                                shipyard_direction = 'SW'
                            if (cardinal_distances[0]<0) and (cardinal_distances[1]<0):
                                shipyard_direction = 'NW'



                if len(enemy_shipyard_locations)<1:
                    enemy_shipyard_distances = [21]
                else:
                    enemy_shipyard_distances = [total_distance_list[ship_location][shipyards] 
                             for shipyards in enemy_shipyard_locations]

                # distance to friendly ships
                friendly_ship_distances = [total_distance_list[ship_location][friendly_location] 
                                           for friendly_location in friendly_ship_locations]

                # distance to enemies with more halite
                fat_enemy_locations = [shipdata[0] for shipdata in enemy_ship_data if shipdata[1]>team_data[2][ship_name][1] ]
                if len(fat_enemy_locations)<1:
                    fat_enemy_distances = [21]
                else:
                    fat_enemy_distances = [total_distance_list[ship_location][fat_enemy_location] 
                                           for fat_enemy_location in fat_enemy_locations]

                # distance to enemy ships with less halite
                lean_enemy_locations = [shipdata[0] for shipdata in enemy_ship_data if shipdata[1]<=team_data[2][ship_name][1] ]
                if len(lean_enemy_locations)<1:
                    lean_enemy_distances = [21]
                else:
                    lean_enemy_distances = [total_distance_list[ship_location][lean_enemy_location]
                                           for lean_enemy_location in lean_enemy_locations]
                    
                    
                
                



                # distance to enemy and friendly shipyards
                self.ship_dist_shipyard.append( min(friendly_shipyard_distances) )    
                self.ship_dist_enemy_yard.append(min(enemy_shipyard_distances) )
                self.ship_direction_shipyard.append(shipyard_direction)

                # amount of halite on the board at distance 0,1,2,3,4,5,6
                self.ship_halite_d0.append(int(board_halite_at_distance[0][ship_location]))
                self.ship_halite_d1.append(int(board_halite_at_distance[1][ship_location]))
                self.ship_halite_d2.append(int(board_halite_at_distance[2][ship_location]))
                self.ship_halite_d3.append(int(board_halite_at_distance[3][ship_location]))
                
                # amount of halite in each direction
                self.ship_halite_north.append( new_obs['halite'][destination_cell(ship_location, ( 0,-1))]
                                              +new_obs['halite'][destination_cell(ship_location, ( 0,-2))]
                                              +new_obs['halite'][destination_cell(ship_location, ( 1,-1))]*0.5
                                              +new_obs['halite'][destination_cell(ship_location, (-1,-1))]*0.5)
                self.ship_halite_south.append( new_obs['halite'][destination_cell(ship_location, ( 0, 1))]
                                              +new_obs['halite'][destination_cell(ship_location, ( 0, 2))]
                                              +new_obs['halite'][destination_cell(ship_location, ( 1, 1))]*0.5
                                              +new_obs['halite'][destination_cell(ship_location, (-1, 1))]*0.5)
                self.ship_halite_east.append( new_obs['halite'][destination_cell(ship_location, ( 1, 0))]
                                             +new_obs['halite'][destination_cell(ship_location, ( 2, 0))]
                                             +new_obs['halite'][destination_cell(ship_location, ( 1, 1))]*0.5
                                             +new_obs['halite'][destination_cell(ship_location, ( 1,-1))]*0.5)
                self.ship_halite_west.append( new_obs['halite'][destination_cell(ship_location, (-1, 0))]
                                             +new_obs['halite'][destination_cell(ship_location, (-2, 0))]
                                             +new_obs['halite'][destination_cell(ship_location, (-1, 1))]*0.5
                                             +new_obs['halite'][destination_cell(ship_location, (-1,-1))]*0.5)
                
                

                # amount of friendly ships on board within distance 1,2,3,4
                self.ship_friendly_d1.append( sum([(friendly_ship_distance<=1) 
                                                   for friendly_ship_distance in friendly_ship_distances])-1)
                self.ship_friendly_d2.append( sum([(friendly_ship_distance<=2) 
                                                   for friendly_ship_distance in friendly_ship_distances])-1)
                self.ship_friendly_d3.append( sum([(friendly_ship_distance<=3) 
                                                   for friendly_ship_distance in friendly_ship_distances])-1)
               


                # fat enemy ships present at distance 1,2,3

                self.ship_fat_enemy_d1.append( sum([(fat_enemy_distance<=1) 
                                                   for fat_enemy_distance in fat_enemy_distances]))
                self.ship_fat_enemy_d2.append( sum([(fat_enemy_distance<=2) 
                                                   for fat_enemy_distance in fat_enemy_distances]))
                self.ship_fat_enemy_d3.append( sum([(fat_enemy_distance<=3) 
                                                   for fat_enemy_distance in fat_enemy_distances]))
     


                # fat enemy ships present at distance 1,2,3

                self.ship_lean_enemy_d1.append( sum([(lean_enemy_distance<=1) 
                                                   for lean_enemy_distance in lean_enemy_distances]))
                self.ship_lean_enemy_d2.append( sum([(lean_enemy_distance<=2) 
                                                   for lean_enemy_distance in lean_enemy_distances]))
                self.ship_lean_enemy_d3.append( sum([(lean_enemy_distance<=3) 
                                                   for lean_enemy_distance in lean_enemy_distances]))
                self.ship_lean_enemy_d4.append( sum([(lean_enemy_distance<=4) 
                                                   for lean_enemy_distance in lean_enemy_distances]))

                if closest_shipyard is None:
                    self.ship_lean_blocking.append(0)
                else:
                    shipyard_path_cells = cells_in_path(ship_location, closest_shipyard)
                    self.ship_lean_blocking.append( sum([lean_enemy_location in shipyard_path_cells 
                                                         for lean_enemy_location in lean_enemy_locations]))

                
                
                
                # pattern of ships within distance of 4 moves
                self.ship_pattern2d.append(
                    nearby_feature_code2d(nearby_features(ship_name, team_num, new_obs, max_distance=2)))
                
                

    def __repr__(self):
        outstring = 'class:ShipOutputGame \n'
        if len(self.ship_list) < 12:
            for key,value in self.__dict__.items():
                outstring += key + ': ' + str(value) + '\n'
        else:
            for key,value in self.__dict__.items():
                outstring += key + ': ' + str(len(value)) + ' values \n'
        return outstring

    
    
def create_ship_game_features(new_obs, cumulative_data):
    # Creates a dictionary of step features where each key is a column name and each value is an array with entries for every ship.
    # This is intended to be used to make an easy dataframe with all or some of the features

    step_num = new_obs['step']
    
    # Output data objects
    board_output = BoardOutputGame(new_obs)
    team_output = TeamOutputGame(new_obs)
    ship_output = ShipOutputGame(new_obs)
    
    # number of ships/observations for this step
    nobs = len(ship_output.ship_list)
    
    # create lists of actions
    action_previous = []
    action_2steps_ago = []
    action_3steps_ago = []
    
    for idx in range(len(ship_output.ship_list)):
        if ship_output.ship_list[idx] in cumulative_data.team_actions_previous[ship_output.ship_team_num[idx]]:
            action_previous.append(cumulative_data.team_actions_previous[ship_output.ship_team_num[idx]][ship_output.ship_list[idx]])
        else:
            action_previous.append(None)
        if ship_output.ship_list[idx] in cumulative_data.team_actions_2steps_ago[ship_output.ship_team_num[idx]]:
            action_2steps_ago.append(cumulative_data.team_actions_2steps_ago[ship_output.ship_team_num[idx]][ship_output.ship_list[idx]])
        else:
            action_2steps_ago.append(None)
        if ship_output.ship_list[idx] in cumulative_data.team_actions_3steps_ago[ship_output.ship_team_num[idx]]:
            action_3steps_ago.append(cumulative_data.team_actions_3steps_ago[ship_output.ship_team_num[idx]][ship_output.ship_list[idx]])
        else:
            action_3steps_ago.append(None)
    
    
    ship_features = {'ship_id':                   ship_output.ship_list,
                     'ship_location':             ship_output.ship_location,
                     'step':                      [step_num]*nobs,
                     'team_num':                  ship_output.ship_team_num,
                     'board_halite_total':        [board_output.board_halite_total]*nobs,
                     'board_ship_total':          [board_output.board_ship_total]*nobs,

                     'team_halite':               [team_output.team_halite[team_num] for team_num in ship_output.ship_team_num],
                     'team_cargo':                [team_output.team_cargo[team_num] for team_num in ship_output.ship_team_num],
                     'team_ships':                [team_output.team_ships[team_num] for team_num in ship_output.ship_team_num],
                     'team_ship_births':          [cumulative_data.team_ship_births[team_num] for team_num in ship_output.ship_team_num],
                     'team_ship_deaths':          [cumulative_data.team_ship_deaths[team_num] for team_num in ship_output.ship_team_num],
                     'team_shipyards':            [team_output.team_shipyards[team_num] for team_num in ship_output.ship_team_num],
                     'team_shipyard_births':      [cumulative_data.team_shipyard_births[team_num] for team_num in ship_output.ship_team_num],
                     'team_shipyard_deaths':      [cumulative_data.team_shipyard_deaths[team_num] for team_num in ship_output.ship_team_num],
                     'team_enemy_ship_kills':     [cumulative_data.team_ship_kills[team_num] for team_num in ship_output.ship_team_num],
                     'team_enemy_shipyard_kills': [cumulative_data.team_shipyard_kills[team_num] for team_num in ship_output.ship_team_num],
                    
                     'ship_cargo':                ship_output.ship_cargo,
                     'ship_dist_shipyard':        ship_output.ship_dist_shipyard,
                     'enemies_blocking_shipyard': ship_output.ship_lean_blocking,
                     'dist_enemy_shipyard':       ship_output.ship_dist_enemy_yard,
                     'halite_dist_0':             ship_output.ship_halite_d0,
                     'halite_dist_1':             ship_output.ship_halite_d1,
                     'halite_dist_3':             ship_output.ship_halite_d3,
                     
                     'direction_shipyard':        ship_output.ship_direction_shipyard,
                     'halite_north':              ship_output.ship_halite_north,
                     'halite_south':              ship_output.ship_halite_south,
                     'halite_east':               ship_output.ship_halite_east,
                     'halite_west':               ship_output.ship_halite_west,
                     
                     
                     
                     'friendly_ships_dist_1':     ship_output.ship_friendly_d1,
                     'friendly_ships_dist_2':     ship_output.ship_friendly_d2,
                     'friendly_ships_dist_3':     ship_output.ship_friendly_d3,
                     'fat_enemy_ships_dist_1':    ship_output.ship_fat_enemy_d1,
                     'fat_enemy_ships_dist_2':    ship_output.ship_fat_enemy_d2,
                     'fat_enemy_ships_dist_3':    ship_output.ship_fat_enemy_d3,
                     'lean_enemy_ships_dist_1':   ship_output.ship_lean_enemy_d1,
                     'lean_enemy_ships_dist_2':   ship_output.ship_lean_enemy_d2,
                     'lean_enemy_ships_dist_3':   ship_output.ship_lean_enemy_d3,
                     'lean_enemy_ships_dist_4':   ship_output.ship_lean_enemy_d4,
                     'local_pattern_dist_2':      ship_output.ship_pattern2d,
                     'action_previous':           action_previous,
                     'action_2steps_ago':         action_2steps_ago,
                     'action_3steps_ago':         action_3steps_ago}
    
    
    return ship_features                


def action_feature_game(ship_data):
    # creates new columns for feature engineering prior to estimating ship value model
    df = ship_data.copy()
    start_cols = df.columns
    
    # Number of steps
    df['step_adj'] = (df['step']>1)*1+(df['step']>25)*1+(df['step']>50)*1+(df['step']>100)*1+(df['step']-370)*(df['step']>370)
    
    
    # ship actions
    df['action_1']= [action_map[string_action] for string_action in df['action_previous']]
    df['action_2']= [action_map[string_action] for string_action in df['action_2steps_ago']]
    df['action_3']= [action_map[string_action] for string_action in df['action_3steps_ago']]
    
    # halite by direction
    df['halite_N'] = df['halite_north'] // 10
    df['halite_S'] = df['halite_south'] // 10
    df['halite_E'] = df['halite_east'] // 10
    df['halite_W'] = df['halite_west'] // 10
    df['halite_X'] = df['halite_dist_0'] // 10
    df['direction_max_halite'] = np.argmax(df[['halite_dist_0','halite_north','halite_south','halite_east','halite_west']].to_numpy(),axis=1)
    
    # distances to bases
    df['dist_friendly_base'] = df['ship_dist_shipyard']*1
    df['dist_enemy_base'] = df['dist_enemy_shipyard']*1
    df['directions_friendly_base']= [two_direction_map[string_action] for string_action in df['direction_shipyard']]
    
    # current cargo and halite
    df['cargo'] = df['ship_cargo'] // 10
    df['team_xs_halite'] = (df['team_halite']>500)*1
    df['no_shipyards'] = (df['team_shipyards'] ==0)*1
    df['one_shipyard'] = (df['team_shipyards'] ==1)*1
    
    # propensity to attack
    df['enemy_kill_rate'] = ((400* df['team_enemy_ship_kills']) // (1+ df['step']))
    df['enemy_attack_d1']= df['fat_enemy_ships_dist_1']
    df['enemy_attack_d2'] = df['fat_enemy_ships_dist_2']
     
    # need to evade
    df['enemy_evade_d1'] = df['lean_enemy_ships_dist_1']
    df['enemy_evade_d2'] = df['lean_enemy_ships_dist_2']
    
    # nearby patterns
    tmp = pattern2_action_probabilities(df['local_pattern_dist_2'])
    df['pattern_X'] = (tmp[:,0] // 0.01 + 1).astype('uint8')
    df['pattern_N'] = (tmp[:,1] // 0.01 + 1).astype('uint8')
    df['pattern_S'] = (tmp[:,2] // 0.01 + 1).astype('uint8')
    df['pattern_E'] = (tmp[:,3] // 0.01 + 1).astype('uint8')
    df['pattern_W'] = (tmp[:,4] // 0.01 + 1).astype('uint8')
    df['pattern_C'] = (tmp[:,5] // 0.01 + 1).astype('uint8')
    

    df.drop(columns=start_cols,inplace=True)
    df = df.astype('uint8')

    return df


def expected_ship_life_pct(game_step, lean_enemy_ships_dist_4, team_ships, board_ship_total, team_ship_births, 
                           team_ship_deaths,ship_dist_shipyard, enemies_blocking_shipyard, three_turn_mortality,
                          scale=1.0):
    # This function returns the expected life of a Halite ship as a percent of steps remaining
    # The parameters in this model are calibrated from 20,000 matches played by top Halite teams
    # resulting in 500 million observations of individual ships at each step in the match
    
    # to avoid problems with divide by zero
    game_step = game_step + ((game_step==0)*1)
    team_ships = team_ships + ((team_ships==0)*1)
    board_ship_total = board_ship_total + ((board_ship_total<4)*(board_ship_total-4))
    
    
    life_pct = (68.2612850844538 + 
                0.09051800214188128 * game_step +
                -0.1583868002150317 * game_step * ((game_step<=25)*1) +
                -0.040762551775483075 * game_step * ((game_step>25)*1)*((game_step<=100)*1) +
                -0.008176527006727103 * game_step * ((game_step>350)*1) * ((game_step<=380)*1) +
                -0.006646679867593494 * game_step * ((game_step>380)*1) +
                -1.4599009883693312 * lean_enemy_ships_dist_4 + 
                0.7310399846970123 * team_ships +
                -9.962766202780374 * ((team_ships==1)*1) +
                0.7110220420296273 * team_ships * ((team_ships>2)*1) * ((team_ships<=30)*1) +
                18.63633230511472 * ((team_ships>30)*1) +
                -0.18107048540612464 * board_ship_total +
                -23.245258831384078 * (team_ships / board_ship_total) +
                -0.432703500440182 * team_ship_births +
                1.8711053233880763 * team_ship_deaths +
                -587.9317730281389 * (team_ship_deaths / game_step) +
                -0.29173115873984856 * ship_dist_shipyard +
                0.21008984344430562 * ship_dist_shipyard * ((ship_dist_shipyard>15)*1) +
                -5.47786457084537 * ((ship_dist_shipyard==21)*1) +
                -64.72313922586982 * three_turn_mortality
               )
    life_pct = 76.0 + scale*(life_pct -76.0) 
    life_pct = life_pct + (100-life_pct) * ((life_pct>100)*1)
    life_pct = life_pct + (0-life_pct) * ((life_pct<0)*1)
    return life_pct


def unresolved_prediction_map(predicted_actions,state,min_probability=0.05):
    
    ship_prediction_map = [[] for idx in range((21**2))]
    
    # place friendly shipyards on the map
    for shipyard in state.shipyards:
        ship_prediction_map[shipyard.location].append(shipyard)
        
    # place enemy shipyards on the map
    for shipyard in state.enemy_shipyards:
        ship_prediction_map[shipyard.location].append(shipyard)
        
    
    # build out a ship prediction map for all ships based on the probabilities
    for ship_idx in range(len(state.ship_data)):
    
        ship_action_probabilities = predicted_actions[ship_idx].copy()
        
        # remove possibilities that are less than min_probability
        ship_action_probabilities[ship_action_probabilities<min_probability] = 0
        if sum(ship_action_probabilities)>0:
            ship_action_probabilities = ship_action_probabilities / sum(ship_action_probabilities)
        else:
            ship_action_probabilities = [p*0 for p in ship_action_probabilities]
        
        for move in range(5):
            if ship_action_probabilities[move]>0:
                new_location = next_location(state.ship_data['ship_location'].iloc[ship_idx],action_map_reverse[move])
                ship_prediction_map[new_location].append(Ship(ship_id = state.ship_data['ship_id'].iloc[ship_idx], 
                                                              location=new_location, 
                                                              halite=state.ship_data['ship_cargo'].iloc[ship_idx],
                                                              team=state.ship_data['team_num'].iloc[ship_idx], 
                                                              probability=ship_action_probabilities[move]))
                
        if ship_action_probabilities[5]>0:
            # it is predicted that the ship will convert to a shipyard
            new_location = state.ship_data['ship_location'].iloc[ship_idx]
            ship_prediction_map[new_location].append(Shipyard(shipyard_id = state.ship_data['ship_id'].iloc[ship_idx], 
                                                              location=new_location,
                                                              team=state.ship_data['team_num'].iloc[ship_idx], 
                                                              probability=ship_action_probabilities[5]))
            
                
    # spawn new enemy ships if there is enough halite and no conflict with other ships
    tmp_team_halite = [state.obs['players'][team_num][0] for team_num in range(4)]
    for shipyard in state.enemy_shipyards:
        # does not currently assume that friendly shipyards will spawn...should it?
        # assume the team will always spawn a new ship if they have more than 500 halite. This is a conservative guess
        # intended to affect the decision on whether or not to attack a shipyard
        if tmp_team_halite[shipyard.team]>500:
            prob_of_same_team_ship = sum([ship.probability for ship in ship_prediction_map[shipyard.location] 
                                        if (ship.team == shipyard.team) and (ship.ship_id != shipyard.ship_id)])
            
            if prob_of_same_team_ship < (1.0-min_probability):
                ship_prediction_map[shipyard.location].append(Ship(ship_id = 'New Ship', 
                                                              location=shipyard.location, 
                                                              halite=0,
                                                              team=shipyard.team, 
                                                              probability=(1.0-prob_of_same_team_ship)))
                tmp_team_halite[shipyard.team] -= 500 
                
    return ship_prediction_map


def update_unresolved_prediction_map(unresolved_map, ship_object, start_location):
    #take locations for ship_id and replace them with actual location (assuming we know with certainty)
    #typically going to be used for own team's ships
    change_cells = cells_in_distance(start_location,1)
    for cell in change_cells:
        relevant_idx = [idx for idx in range(len(unresolved_map[cell]))
                        if unresolved_map[cell][idx].ship_id == ship_object.ship_id]
        if len(relevant_idx)>0:
            unresolved_map[cell].pop(relevant_idx[0])
            
    unresolved_map[ship_object.location].append(ship_object)
    
    return unresolved_map


def resolve_prediction_map(unresolved_map, ship_object, end_location, state):
    # initialize resolved map
    ship_resolved_prediction_map = [[] for idx in range((21**2))]
    expected_enemy_ship_kill = 0.0
    expected_enemy_shipyard_kill = 0.0
    
    # move ship to assumed location
    start_location = ship_object.location
    ship_object.location = end_location
    unresolved_map = update_unresolved_prediction_map(unresolved_map, ship_object, start_location)
    
    ############################
    ##
    ## For debugging
    #if ship_object.ship_id == '9-1':
    #    print('For location: ' + str(end_location))
    #    print("Ship object input is:")
    #    print(ship_object)
    #    print("and the updated unresolved map at that location is:")
    #    print(unresolved_map[end_location])
    
    #############################
    
    for cell_index, cell in enumerate(unresolved_map):
        #for each cell resolve potential collissions
        if len(cell)==1:
            ship_resolved_prediction_map[cell_index] = cell
            if cell[0].ship_id == ship_object.ship_id:
                if type(ship_object).__name__ == 'Ship':
                    resolved_ship_object = Ship(ship_id = cell[0].ship_id,
                                                location = cell_index,
                                                halite = cell[0].halite,
                                                team = cell[0].team,
                                                probability = cell[0].probability)
                elif type(ship_object).__name__ == 'Shipyard':
                    resolved_ship_object = Shipyard(shipyard_id = cell[0].shipyard_id,
                                                    location = cell_index,
                                                    team = cell[0].team,
                                                    probability = cell[0].probability)
                    
            
        elif len(cell)>1:
            ship_teams = [ship.team for ship in cell]
            leanest_ship_team_prob = np.array([0.0,0.0,0.0,0.0])
            expected_conquest = 0.0
            expected_ship_kills = 0.0
            if max(ship_teams)==min(ship_teams):
                # all same team, assume cooperative
                probability_sorter = [ [(-1.0*cell[idx].probability),idx] if type(cell[idx]).__name__=='Ship' else [9999,idx] for idx in range(len(cell))]
                probability_sorter.sort()
                cell_ship_order = [ship_vals[1] for ship_vals in probability_sorter]
                friendly_cumulative_probability = 0.0
                

                for ship_idx in cell_ship_order:
                    # check if this is a ship or a shipyard
                    if type(cell[ship_idx]).__name__=='Ship':
                        adj_probability = min(cell[ship_idx].probability,1.0-friendly_cumulative_probability)
                        friendly_cumulative_probability += adj_probability
                        if (adj_probability>0) or (cell[ship_idx].ship_id == ship_object.ship_id):
                            ship_resolved_prediction_map[cell_index].append(Ship(ship_id = cell[ship_idx].ship_id,
                                                                                 location = cell[ship_idx].location,
                                                                                 halite = cell[ship_idx].halite,
                                                                                 team = cell[ship_idx].team,
                                                                                 probability = adj_probability))
                            
                        if cell[ship_idx].ship_id == ship_object.ship_id:
                            resolved_ship_object = Ship(ship_id = cell[ship_idx].ship_id,
                                                        location = cell_index,
                                                        halite = cell[ship_idx].halite,
                                                        team = cell[ship_idx].team,
                                                        probability = adj_probability)
                        
                        
                    elif type(cell[ship_idx]).__name__=='Shipyard':
                        ship_resolved_prediction_map[cell_index].append(Shipyard(shipyard_id = cell[ship_idx].shipyard_id,
                                                                                 location = cell[ship_idx].location,
                                                                                 team = cell[ship_idx].team,
                                                                                 probability = cell[ship_idx].probability))
                        if cell[ship_idx].shipyard_id == ship_object.ship_id:
                            resolved_ship_object = Shipyard(shipyard_id = cell[ship_idx].shipyard_id,
                                                        location = cell_index,
                                                        team = cell[ship_idx].team,
                                                        probability = adj_probability)
                        
                        
            else:
                # there is at least one collission across different teams
                halite_sorter = [ [cell[idx].halite,idx] if type(cell[idx]).__name__=='Ship' else [9999,idx] for idx in range(len(cell))]
                halite_sorter.sort()
                cell_ship_order = [ship_vals[1] for ship_vals in halite_sorter]
                tmp_leanest_ship_team_prob = np.array([0.0,0.0,0.0,0.0])
                for ship_idx in cell_ship_order:
                    # check if this is a ship or a shipyard
                    if type(cell[ship_idx]).__name__=='Ship':
                        
                        adj_probability = min(cell[ship_idx].probability,1-leanest_ship_team_prob[cell[ship_idx].team])
                        
                        # initialize probability of outcome destroyed
                        prob_outcome_destroyed = 0.0
                        
                        # probability of being destroyed in a tie
                        any_ties = False
                        tied_halite_team_probabilities = np.array([0.0,0.0,0.0,0.0])
                        for other_ship in cell:
                            if type(other_ship).__name__=='Ship' and other_ship.halite == cell[ship_idx].halite and other_ship.ship_id != cell[ship_idx].ship_id:
                                any_ties = True
                                tied_halite_team_probabilities[other_ship.team] += other_ship.probability
                        tie_survival = 1.0
                        for team_num in range(4):
                            if team_num != cell[ship_idx].team:
                                tie_survival = tie_survival*(1-tied_halite_team_probabilities[team_num])
                        prob_outcome_destroyed += (adj_probability-prob_outcome_destroyed)*(1-tie_survival)
                        
                        # updating the leanest ship team probabilities of not tied with others
                        if any_ties:
                            tmp_leanest_ship_team_prob[cell[ship_idx].team] += (adj_probability - prob_outcome_destroyed)
                        else:
                            leanest_ship_team_prob += tmp_leanest_ship_team_prob
                            tmp_leanest_ship_team_prob = np.array([0.0,0.0,0.0,0.0])
                        

                        # probability of being destroyed by stronger ships
                        same_team_leaner = leanest_ship_team_prob[cell[ship_idx].team]
                        other_team_leaner = sum(leanest_ship_team_prob) - same_team_leaner
                        prob_outcome_destroyed += adj_probability*other_team_leaner
                        tmp_leanest_ship_team_prob[cell[ship_idx].team] += (adj_probability - prob_outcome_destroyed)
                        
                        #if ship_object.ship_id == '9-1':
                            #print("PROBABILITY OF BEING DESTROYED")
                            #print("same_team_leaner: " + str(same_team_leaner))
                            #print("other_team_leaner: " + str(other_team_leaner))
                            #print("prob_outcome_destroyed:" + str(prob_outcome_destroyed))   
                            #print("tmp_leanest_ship_team_prob:")
                            #print(tmp_leanest_ship_team_prob)
                        
                        # being destroyed by shipyard collission
                        for other_ship in cell:
                            if type(other_ship).__name__=='Shipyard' and other_ship.team != cell[ship_idx].team:
                                # note shipyard kill of this is the ship of interest
                                if cell[ship_idx].ship_id == ship_object.ship_id:
                                    expected_enemy_shipyard_kill = 0.0 + (adj_probability-prob_outcome_destroyed)*other_ship.probability
                                # ship is destroyed in enemy shipyard collission
                                prob_outcome_destroyed += (adj_probability-prob_outcome_destroyed)*other_ship.probability
                                
                                
                        # actually surving as the leanest ship
                        if prob_outcome_destroyed < adj_probability:
                            expected_conquest = 0.0
                            expected_ship_kills = 0.0
                            for other_ship in cell:
                                if type(other_ship).__name__=='Ship' and other_ship.team != cell[ship_idx].team:
                                    if other_ship.halite > cell[ship_idx].halite:
                                        expected_conquest += other_ship.halite*other_ship.probability
                                        expected_ship_kills += other_ship.probability
                                        
                            ship_resolved_prediction_map[cell[ship_idx].location].append(Ship(ship_id = cell[ship_idx].ship_id,
                                                                                         location = cell[ship_idx].location,
                                                                                         halite = cell[ship_idx].halite + expected_conquest,
                                                                                         team = cell[ship_idx].team,
                                                                                         probability = adj_probability - prob_outcome_destroyed))
                                
                        
                        if cell[ship_idx].ship_id == ship_object.ship_id:
                            expected_enemy_ship_kill = 0.0 + expected_ship_kills
                            resolved_ship_object = Ship(ship_id = cell[ship_idx].ship_id,
                                                        location = cell[ship_idx].location,
                                                        halite = cell[ship_idx].halite + expected_conquest,
                                                        team = cell[ship_idx].team,
                                                        probability = adj_probability - prob_outcome_destroyed)
                            ### DEBUGGING #####
                            #if ship_object.ship_id == '9-1':
                            #    print("IN RESOLUTION STEP....")
                            #    print(resolved_ship_object)
                                
                                
                    
                    elif type(cell[ship_idx]).__name__=='Shipyard':
                        leanest_ship_team_prob += tmp_leanest_ship_team_prob
                        shipyard_survival_prob = 1-(sum(leanest_ship_team_prob) - leanest_ship_team_prob[cell[ship_idx].team])
                        # probability of no enemy ship surviving
                        if shipyard_survival_prob>0:
                            ship_resolved_prediction_map[cell[ship_idx].location].append(Shipyard(shipyard_id = cell[ship_idx].shipyard_id,
                                                                                             location = cell[ship_idx].location,
                                                                                             team = cell[ship_idx].team,
                                                                                             probability = cell[ship_idx].probability*shipyard_survival_prob))
                            
                        if cell[ship_idx].shipyard_id == ship_object.ship_id:
                            resolved_ship_object = Shipyard(ship_id = cell[ship_idx].shipyard_id,
                                                        location = cell[ship_idx].location,
                                                        team = cell[ship_idx].team,
                                                        probability = adj_probability - prob_outcome_destroyed)
                            
    # next turn resolution steps are
    #- deposit halite
    expected_team_halite = [ player_data[0] for player_data in state.obs['players']]
    for cell_index, cell in enumerate(ship_resolved_prediction_map):
        if len(cell)>1:
            for check_object in cell:
                if type(check_object).__name__=='Shipyard':
                    # potential for ships to deposit halite
                    for ship_index, ship_object in enumerate(cell):
                        if type(ship_object).__name__=='Ship' and ship_object.team==check_object.team:
                            expected_deposit = ship_resolved_prediction_map[cell_index][ship_index].halite*min(check_object.probability,ship_object.probability)
                            expected_team_halite[check_object.team] += expected_deposit
                            ship_resolved_prediction_map[cell_index][ship_index].halite = ship_resolved_prediction_map[cell_index][ship_index].halite*check_object.probability
                            
    
    #- mine halite
    expected_board_halite = state.obs['halite'].copy()
    for team_num in range(4):
        for ship_id in state.obs['players'][team_num][2].keys():
            prev_location = state.obs['players'][team_num][2][ship_id][0]
            ships_in_cell = [other_ship.ship_id for other_ship in ship_resolved_prediction_map[prev_location]]
            if ship_id in ships_in_cell:
                if type([ships_in_cell.index(ship_id)]).__name__ == 'Ship':
                    # ship stayed in same location and would be mining
                    ship_resolved_prediction_map[prev_location][ships_in_cell.index(ship_id)].halite += 0.25*state.obs['halite'][prev_location]
                    expected_board_halite[prev_location] -= 0.25*state.obs['halite'][prev_location]*ship_resolved_prediction_map[prev_location][ships_in_cell.index(ship_id)].probability
    
    #- regenerate halite
    for cell_index in range(len(expected_board_halite)):
        expected_board_halite[cell_index] = min(500, 1.02*expected_board_halite[cell_index])
    
    return resolved_ship_object, ship_resolved_prediction_map, expected_board_halite, expected_team_halite, expected_enemy_ship_kill, expected_enemy_shipyard_kill



def resolved_pattern_probabilities(ship_object, ship_resolved_prediction_map, min_probability=0.1):
    
    ordered_cells = cells_in_distance(location=ship_object.location,max_distance=2)

    feature_arrays = [[2]]
    probabilities = [1.0]

    for cell in ordered_cells[1:]:
        
        cell_feature = [0,1,2,3,4,5]
        cell_feature_probabilities = [0.0,0.0,0.0,0.0,0.0,0.0,]
        probable_ships = [ship for ship in ship_resolved_prediction_map[cell] if ship.probability>min_probability]

        if len(probable_ships)>0:     
            for ship in probable_ships:
                # feature 0: empty cell
                # feature 1: friendly shipyard
                # feature 2: friendly ship
                # feature 3: enemy shipyard
                # feature 4: fat enemy ship
                # feature 5: lean enemy ship
                
                if type(ship).__name__ == 'Shipyard' and ship.team == ship_object.team:
                    cell_feature_probabilities[1] += ship.probability
                elif type(ship).__name__ == 'Ship' and ship.team == ship_object.team:
                    cell_feature_probabilities[2] += ship.probability
                elif type(ship).__name__ == 'Shipyard' and ship.team != ship_object.team:
                    cell_feature_probabilities[3] += ship.probability
                elif type(ship).__name__ == 'Ship' and ship.team != ship_object.team:
                    if ship.halite > ship_object.halite:
                        cell_feature_probabilities[4] += ship.probability
                    else:
                        cell_feature_probabilities[5] += ship.probability
                # if both shipyard and ships, count the ships
                if cell_feature_probabilities[1]>0 and sum(cell_feature_probabilities)>1.0:
                    cell_feature_probabilities[1] = max(0.0,(cell_feature_probabilities[1] - (sum(cell_feature_probabilities)-1)))
                if cell_feature_probabilities[3]>0 and sum(cell_feature_probabilities)>1.0:
                    cell_feature_probabilities[3] = max(0.0,(cell_feature_probabilities[3] - (sum(cell_feature_probabilities)-1)))
                    
        # assign remaining probability to empty cell
        cell_feature_probabilities[0] = 1.0 - sum(cell_feature_probabilities)
                    
        for feature_index in [5,4,3,2,1,0]:
            if cell_feature_probabilities[feature_index]==0.0:
                del cell_feature[feature_index]
                del cell_feature_probabilities[feature_index]
            
        # - add nodes to tree
        old_feature_arrays = copy.copy(feature_arrays)
        old_probabilities = copy.copy(probabilities)

        feature_arrays = []
        probabilities = []
        for idx in range(len(cell_feature)):
            feature_arrays += [old_feature_array+[cell_feature[idx]] for old_feature_array in old_feature_arrays]
            probabilities += [pattern_probability*cell_feature_probabilities[idx] for pattern_probability in old_probabilities ]
    
    feature_codes = [0]*len(feature_arrays)
    for outcome_num in range(len(feature_arrays)):

        feature_codes[outcome_num] = int(nearby_feature_code2d(feature_arrays[outcome_num]))
        
    return feature_codes, probabilities


def value_feature_fast(ev_data_dict):
    
    N = len(ev_data_dict['game_step'])
    # preallocate output
    vf = np.zeros((N,119))
    
    # 1 - constant
    vf[:,0] = 1
    
    steps_left = 399 - ev_data_dict['game_step']
    # Expected Life
    exp_life = steps_left*ev_data_dict['exp_life_pct'] 
    total_halite_by_life = ev_data_dict['board_halite_total']*exp_life
    
    
    ## Current Cargo
    for sy_dist in range(1,21):
        vf[:,sy_dist] = ev_data_dict['ship_cargo'] * (ev_data_dict['ship_dist_shipyard']==sy_dist) * (steps_left>=ev_data_dict['ship_dist_shipyard'])
        
    # cargo_distX
    vf[:,21] = ev_data_dict['ship_cargo'] * (ev_data_dict['ship_dist_shipyard']==21) * (steps_left>=ev_data_dict['ship_dist_shipyard'])
    # cargo_xs500
    vf[:,22] = (ev_data_dict['ship_cargo']-500) * (ev_data_dict['ship_cargo']>500) * (steps_left<ev_data_dict['ship_dist_shipyard'])
    # cargo_blocked
    vf[:,23] = ev_data_dict['ship_cargo'] * ((ev_data_dict['enemies_blocking_shipyard']>0)*1)
    
    
    # Piecewise separation for total amount of halite
    vf[:,24] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life<1e+07)
    vf[:,25] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=1e+07)*(total_halite_by_life<2e+07)
    vf[:,26] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=2e+07)*(total_halite_by_life<3e+07)
    vf[:,27] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=3e+07)*(total_halite_by_life<5e+07)
    vf[:,28] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=5e+07)*(total_halite_by_life<1.3e+08)
    vf[:,29] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=1.3e+08)*(total_halite_by_life<1.7e+08)
    vf[:,30] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=1.7e+08)*(total_halite_by_life<2.1e+08)
    vf[:,31] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=2.1e+08)*(total_halite_by_life<2.6e+08)
    vf[:,32] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=2.6e+08)*(total_halite_by_life<3.1e+08)
    vf[:,33] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=3.1e+08)*(total_halite_by_life<4.0e+08)
    vf[:,34] = (total_halite_by_life*(1e-06) - 150)*(total_halite_by_life>=4.0e+08)
    
    
    # Total halite interracted with number of ships
    for nships in range(40,91):
        vf[:,nships-5] = (total_halite_by_life*(1e-06) - 150)*(ev_data_dict['board_ship_total']==nships)
        
    vf[:,86] = (total_halite_by_life*(1e-06) - 150)*(ev_data_dict['board_ship_total']>90)

    # Total halite interracted with number of friendly ships
    for nships in range(10,33):
        vf[:,77+nships] = (total_halite_by_life*(1e-06) - 150)*(ev_data_dict['team_ships']==nships)
        
    vf[:,110] = (total_halite_by_life*(1e-06) - 150)*(ev_data_dict['team_ships']>32)
    
    # Local Region
    # h0
    vf[:,111] = ev_data_dict['halite_dist_0'] * (1 - ev_data_dict['three_turn_mortality'])
    # h3_near
    vf[:,112] = (ev_data_dict['halite_dist_3'] * (ev_data_dict['exp_life_pct'])*0.01) * (ev_data_dict['ship_dist_shipyard']<5)
    # h3_mid
    vf[:,113] = (ev_data_dict['halite_dist_3'] * (ev_data_dict['exp_life_pct'])*0.01) * (ev_data_dict['ship_dist_shipyard']>=5) * (ev_data_dict['ship_dist_shipyard']<10)
    # h3_far
    vf[:,114] = (ev_data_dict['halite_dist_3'] * (ev_data_dict['exp_life_pct'])*0.01) * (ev_data_dict['ship_dist_shipyard']>=10) * (ev_data_dict['ship_dist_shipyard']<=16)
    # h3_distant
    vf[:,115] = (ev_data_dict['halite_dist_3'] * (ev_data_dict['exp_life_pct'])*0.01) * (ev_data_dict['ship_dist_shipyard']>16)
    # ships_dist_3
    vf[:,116] = (ev_data_dict['friendly_ships_dist_3'] + ev_data_dict['lean_enemy_ships_dist_3'] + ev_data_dict['fat_enemy_ships_dist_3'])
    # h3_per_ship
    vf[:,117] = ev_data_dict['halite_dist_3'] / (1+(ev_data_dict['friendly_ships_dist_3'] + ev_data_dict['lean_enemy_ships_dist_3'] + ev_data_dict['fat_enemy_ships_dist_3']))
    # h3_blocked
    vf[:,118] = ev_data_dict['halite_dist_3'] * ((ev_data_dict['enemies_blocking_shipyard']>0)*1)
    
    return vf


def expected_values(ship_object, locations, state, turn, unres_map):
    # for a given ship_id, the function calculates the expected value for various locations
    
    # Initialize features
    game_step = []
    board_halite_total = []
    ship_cargo = []
    team_ships = []
    board_ship_total = []
    ship_dist_shipyard = []
    halite_dist_0 = []
    halite_dist_3 = []
    friendly_ships_dist_3 = []
    lean_enemy_ships_dist_3 = []
    fat_enemy_ships_dist_3 = []
    enemies_blocking_shipyard = []
    exp_life_pct = []
    three_turn_mortality = []
    # for expected life calculation
    lean_enemy_ships_dist_4 = []
    team_ship_births = []
    team_ship_deaths = []
    # output of expected life calculation
    exp_life_pct = []
    # output variables
    survival_probability = []
    expected_ship_kills = []
    expected_shipyard_kills= []
    # halite delivery
    exp_next_delivery =[]
    
    
    # make a list of friendly and enemy shipyard locations
    friendly_shipyard_locations = [shipyard.location for shipyard in state.shipyards]
    
    # update for actions already selected
    for team_ship_id in turn.next_actions:
        if turn.next_actions[team_ship_id]=='CONVERT':
            team_ship_object = Shipyard(shipyard_id=team_ship_id,
                                        location = state.obs['players'][state.team][2][team_ship_id][0],
                                        team = state.team,
                                        probability = 1.0)
            unres_map = update_unresolved_prediction_map(unresolved_map=unres_map, 
                                                         ship_object = team_ship_object, 
                                                         start_location = state.obs['players'][state.team][2][team_ship_id][0])
        elif turn.next_actions[team_ship_id]=='SPAWN':
            pass
        else:
            team_ship_object = Ship(ship_id=team_ship_id,
                                    location = next_location(state.obs['players'][state.team][2][team_ship_id][0], turn.next_actions[team_ship_id]),
                                    halite = state.obs['players'][state.team][2][team_ship_id][1],
                                    team = state.team,
                                    probability = 1.0)
            unres_map = update_unresolved_prediction_map(unresolved_map=unres_map, 
                                                         ship_object = team_ship_object, 
                                                         start_location = state.obs['players'][state.team][2][team_ship_id][0])

    # calculate ship-specific variables by resolving halite map assuming ship_object is at each end_location in locations
    for end_location in locations:
        
        game_step.append(state.obs['step'])
        
        new_ship_object = Ship(ship_id=ship_object.ship_id,
                           location = ship_object.location,
                           halite = ship_object.halite,
                           team = ship_object.team,
                           probability = 1.0)
        
        res_ship, res_map, exp_board_halite, exp_team_halite, exp_next_kill, exp_next_shipyard = resolve_prediction_map(unres_map, 
                                                                                                                   ship_object=new_ship_object,
                                                                                                                   end_location=end_location,
                                                                                                                   state=state)
        # note halite delivery
        if sum( [ unit_object.probability for unit_object in res_map[end_location] if type(unit_object).__name__=='Shipyard'] ) > 0:
            exp_next_delivery.append(ship_object.halite*res_ship.probability)
            res_ship.halite = 0
        else:
            exp_next_delivery.append(0)
            
        ###################################
        # Print out for debugging
        #if res_ship.ship_id == "9-1":
        #    print(res_ship.ship_id + "at location " + str(end_location) + ": " + str(res_ship.probability) + " survival")
        ##################################
        survival_probability.append(res_ship.probability )
        board_halite_total.append( sum(exp_board_halite))
        
        # change the predicted ship cargo from the prediction values to an updated value
        ship_cargo.append(res_ship.halite)
        team_ships.append( len(state.obs['players'][ship_object.team][2]) ) # + new births - new_deaths?
        team_ship_births.append( state.cumulative_data.team_ship_births[ship_object.team] ) # + new births?
        team_ship_deaths.append( state.cumulative_data.team_ship_deaths[ship_object.team] ) # + new deaths?
        board_ship_total.append( sum([len(state.obs['players'][team_num][2]) for team_num in range(4)])) # + new births - new_deaths?
        
        
        # distance to closest shipyard (does not account for newly created shipyards)
        if len(friendly_shipyard_locations)<1:
            friendly_shipyard_distances = [21]
            closest_shipyard = None
            sy_dist = 21
            ship_dist_shipyard.append( sy_dist)
        else:
            friendly_shipyard_distances = [total_distance_list[end_location][shipyards] 
                                            for shipyards in friendly_shipyard_locations]
            sy_dist = min(friendly_shipyard_distances)
            ship_dist_shipyard.append( sy_dist )
            closest_shipyard = friendly_shipyard_locations[friendly_shipyard_distances.index(sy_dist)]
        
        
        halite_dist_0.append( exp_board_halite[end_location])
        
        d3_cells = cells_in_distance(end_location,3)
        d4_cells = cells_in_distance(end_location,4)
        halite_dist_3.append( sum( [exp_board_halite[d3_cell] for d3_cell in d3_cells] ))
        
        fs_dist_3 = 0.0
        le_dist_3 = 0.0
        fe_dist_3 = 0.0
        
        for d3_cell in d3_cells:
            ships_in_cell = [unit for unit in res_map[d3_cell] if type(unit).__name__=='Ship']
            fs_dist_3 += sum([friendly_ship.probability for friendly_ship in ships_in_cell 
                                          if friendly_ship.team==ship_object.team and d3_cell != end_location])
            
            fe_dist_3 += sum([enemy_ship.probability for enemy_ship in ships_in_cell 
                                          if enemy_ship.team!=ship_object.team and enemy_ship.halite>ship_object.halite])
            
            le_dist_3 += sum([enemy_ship.probability for enemy_ship in ships_in_cell 
                                          if enemy_ship.team!=ship_object.team and enemy_ship.halite<=ship_object.halite])
        
        le_dist_4 = le_dist_3*1.0
        for d4_cell in d4_cells:
            if d4_cell not in d3_cells:
                ships_in_cell = [unit for unit in res_map[d4_cell] if type(unit).__name__=='Ship']
                le_dist_4 += sum([enemy_ship.probability for enemy_ship in ships_in_cell 
                                                if enemy_ship.team!=ship_object.team and enemy_ship.halite<=ship_object.halite])
        
        friendly_ships_dist_3.append(fs_dist_3)
        lean_enemy_ships_dist_3.append(le_dist_3)
        fat_enemy_ships_dist_3.append(fe_dist_3)
        lean_enemy_ships_dist_4.append(le_dist_4)
        
        blockers = 0.0    
        if closest_shipyard is None:
            pass
        else:
            shipyard_path_cells = cells_in_path(end_location, closest_shipyard)
            for shipyard_path_cell in shipyard_path_cells:
                ships_in_cell = [unit for unit in res_map[shipyard_path_cell] if type(unit).__name__=='Ship']
                blockers += sum([enemy_ship.probability for enemy_ship in ships_in_cell
                                 if enemy_ship.team!=ship_object.team and enemy_ship.halite<=res_ship.halite])
        enemies_blocking_shipyard.append(blockers)
    
        # three turn mortality
        loc_patterns, loc_pattern_probs = resolved_pattern_probabilities(res_ship, res_map, min_probability=0.1)
        three_turn_mortality.append( exp_mortality(loc_patterns, loc_pattern_probs))
        expected_ship_kills.append( exp_enemy_kills(loc_patterns, loc_pattern_probs) + exp_next_kill)
        
        expected_shipyard_kills.append( min(1.0, exp_shipyard_kill(loc_patterns, loc_pattern_probs) + exp_next_shipyard) )

        
    # ship life pct
    exp_life_pct = expected_ship_life_pct(np.array(game_step), np.array(lean_enemy_ships_dist_4), np.array(team_ships), 
                                      np.array(board_ship_total), np.array(team_ship_births), np.array(team_ship_deaths),
                                      np.array(ship_dist_shipyard), np.array(enemies_blocking_shipyard), np.array(three_turn_mortality))
    
    
    value_features =value_feature_fast({'game_step': np.array(game_step),
                                        'board_halite_total': np.array(board_halite_total),
                                        'ship_cargo': np.array(ship_cargo),
                                        'team_ships': np.array(team_ships),
                                        'board_ship_total': np.array(board_ship_total),
                                        'ship_dist_shipyard': np.array(ship_dist_shipyard),
                                        'halite_dist_0': np.array(halite_dist_0),
                                        'halite_dist_3': np.array(halite_dist_3),
                                        'friendly_ships_dist_3': np.array(friendly_ships_dist_3),
                                        'lean_enemy_ships_dist_3': np.array(lean_enemy_ships_dist_3),
                                        'fat_enemy_ships_dist_3': np.array(fat_enemy_ships_dist_3), 
                                        'enemies_blocking_shipyard': np.array(enemies_blocking_shipyard),
                                        'exp_life_pct': np.array(exp_life_pct), 
                                        'three_turn_mortality': np.array(three_turn_mortality) })
    
    expected_halite_delivered = np.squeeze(np.dot(value_features,ev_params))*np.array(survival_probability)
    expected_halite_delivered += np.array(exp_next_delivery)
    
    return expected_halite_delivered, np.array(expected_ship_kills), np.array(expected_shipyard_kills)
    
class KoopaStrategy:
    def __init__(self, state):
        # The Koopa Troopa strategy parameters simply consist of a ship building schedule,
        # probability thresholds for action, and an amount that 
        
        
        self.min_probability = 0.1
        self.good_probability = 0.5
        self.shipyard_min_sacrifice=50 #max halite cargo for shipyard attack
        
        # set schedule for maximum number of ships
        if state.obs['step']<200:
            self.max_ships = 40
        elif state.obs['step']<300:
            self.max_ships = 25
        elif state.obs['step']<350:
            self.max_ships = 20
        elif state.obs['step']<380:
            self.max_ships = 15
        else:
            self.max_ships = 2
            
    def __repr__(self):
        outstring = 'class:Strategy \n'
        for key,value in self.__dict__.items():
            outstring += '  ' + key + ':' + str(value) + '\n'
        return outstring


def random_prediction(state):
    potential_moves = ['NORTH','SOUTH','EAST','WEST',None]
    unresolved_prediction_map = [[Ship(ship_id='Ghost',location=idx,probability=0.0)] for idx in range((21**2))]
    for ship in (state.enemy_ships + state.ships):
        for move in potential_moves:
            probable_ship = copy.copy(ship)
            probable_ship.location = next_location(ship.location,move)
            probable_ship.probability = 0.2
            unresolved_prediction_map[probable_ship.location].append(probable_ship)
    return unresolved_prediction_map


def koopa_convert(state, turn, unresolved_prediction_map, strategy):
    #if no shipyard, convert the ship carrying max halite unless it is in danger
    if len(state.shipyards)==0 and len(state.ships)>0 and state.obs['step']<399:
        # get the ship_id for the ship with the most halite
        mx={ship.halite:ship.ship_id for ship in state.ships}[max([ship.halite for ship in state.ships])]
        if state.obs['players'][state.obs['player']][0] + state.obs['players'][state.obs['player']][2][mx][1] > 500:
            turn.next_actions[mx] = 'CONVERT'
            turn.shipyard_locations[mx] = state.obs['players'][state.obs['player']][2][mx][0]
            turn.team_halite -= 500 - state.obs['players'][state.obs['player']][2][mx][1]
            
    # define the minimum probability for which we want to react to enemy potential presence
    
    for ship in state.ships:
        if ship.ship_id not in turn.next_actions:
            # calculate distance to nearest shipyard
            shipyard_dist,_ = nearest_shipyard(ship.location, team_num = state.obs['player'], obs = state.obs)
            
            # calculate the cargo value of enemy ships within a distance of one
            ship_moves = cells_in_distance(ship.location,1)
            for location in ship_moves:
                probable_enemy_cargos = [other_ship.halite for other_ship in unresolved_prediction_map[location] 
                                         if other_ship.probability>=strategy.min_probability and other_ship.team != state.team]

            # Consider ships that are carrying lots of halite
            if ship.halite>500:
                # assume there is no escape and check if any locations 
                escape_potential = False
                if len(probable_enemy_cargos) < 5:
                    escape_potential = True
                elif min(probable_enemy_cargos)>ship.halite:
                    escape_potential = True
                if escape_potential == False:
                    # [A] convert if there is a greater than min_probability chance of being destroyed in each cell and ship has >500 halite in cargo
                    turn.next_actions[ship.ship_id] = 'CONVERT'
                    turn.shipyard_locations[ship.ship_id] = state.obs['players'][state.obs['player']][2][ship.ship_id][0]
                    turn.team_halite -= 500 - ship.halite
                elif state.obs['step']>396 and shipyard_dist>1:
                    # [B] convert if not possible to return to shipyard by last step
                    turn.next_actions[ship.ship_id] = 'CONVERT'
                    turn.shipyard_locations[ship.ship_id] = state.obs['players'][state.obs['player']][2][ship.ship_id][0]
                    turn.team_halite -= 500 - ship.halite
                    
                    
            #CHECK if we're hauling long distance without threats
            no_enemy_nearby = sum([state.board.enemy_present_map[cell_idx] for cell_idx in cells_in_distance(ship.location,4)])==0
            
            if shipyard_dist>9 and len(state.shipyards)<2 and ship.halite+turn.team_halite > 750 and no_enemy_nearby and state.obs['step'] < 350:
                turn.next_actions[ship.ship_id] = 'CONVERT'
                turn.shipyard_locations[ship.ship_id] = state.obs['players'][state.obs['player']][2][ship.ship_id][0]
                turn.team_halite -= 500 - ship.halite
            if shipyard_dist>13 and len(state.shipyards)<3 and ship.halite+turn.team_halite > 1000 and no_enemy_nearby and state.obs['step'] < 300:
                turn.next_actions[ship.ship_id] = 'CONVERT'
                turn.shipyard_locations[ship.ship_id] = state.obs['players'][state.obs['player']][2][ship.ship_id][0]
                turn.team_halite -= 500 - ship.halite
                    
    return turn
            
            
def koopa_path(ship_objectives,state, turn, unresolved_prediction_map, strategy):
    all_actions = ['NORTH','SOUTH','EAST','WEST',None]
    
    for ship in state.ships:
        if ship.ship_id not in turn.next_actions:
        
            # get the directions that would lead toward the ship's objective
            if ship.ship_id in ship_objectives:
                ship_actions = next_directions(ship.location, ship_objectives[ship.ship_id])
            else:
                ship_actions = [None]
            alt_actions = [action for action in all_actions if action not in ship_actions]

            # dictionary of the ship's next action destinations and the action that takes it there
            ship_destinations = {next_location(ship.location,action):action for action in ship_actions}

            # remove destinations that would be problematic
            tmp_destinations = list(ship_destinations.keys())
            for destination in tmp_destinations:
                # check if another ship is already going to be in that location next turn
                if destination in turn.next_ship_locations:     
                    ship_destinations.pop(destination,None)
                else:
                    # check if an enemy ship with less halite is there with a probability greater than min_probability
                    for other_ship in unresolved_prediction_map[destination]:
                        if other_ship.halite <= ship.halite and other_ship.probability > strategy.min_probability and other_ship.team != state.team:
                            ship_destinations.pop(destination,None)

            # if none of the original potential actions are safe, consider all the alternative actions
            if len(ship_destinations)==0:
                ship_actions = alt_actions
                ship_destinations = {next_location(ship.location,action):action for action in ship_actions}
                tmp_destinations = list(ship_destinations.keys())
                for destination in tmp_destinations:
                    if destination in turn.next_ship_locations:
                        ship_destinations.pop(destination,None)
                    else:
                        for other_ship in unresolved_prediction_map[destination]:
                            if other_ship.halite <= ship.halite and other_ship.probability > strategy.min_probability and other_ship.team != state.team:
                                ship_destinations.pop(destination,None)

            # if absolutely no option is safe, take the best of the bad options 
            if len(ship_destinations)==0:
                ship_actions = copy.copy(all_actions)
                ship_destinations = {next_location(ship.location,action):action for action in ship_actions}
                lowest_probability = 2.0
                tmp_destinations = list(ship_destinations.keys())
                for destination in tmp_destinations:
                    worst_probability = 0.0
                    for other_ship in unresolved_prediction_map[destination]:
                        if other_ship.halite < ship.halite and other_ship.probability > worst_probability and other_ship.team != state.team:
                            worst_probability = other_ship.probability+0.0

                    if worst_probability < lowest_probability:
                        lowest_probability = worst_probability+0.0
                    else:
                        ship_destinations.pop(destination,None)

                # at this point, there should be at least one value in the ship_destinations dictionary

            # check if attractive attacks are available
            adjacent_cells = cells_in_distance(ship.location, max_distance=1)
            
            attack_probability = 0.0
            for adjacent_cell in adjacent_cells:
                # Look at possibility of attacking enemy shipyards
                for enemy_shipyard in state.enemy_shipyards:
                    if ship.ship_id not in turn.next_actions:
                        if enemy_shipyard.location in adjacent_cells:
                            if ship.halite<strategy.shipyard_min_sacrifice:
                                if len([other_ship for other_ship in unresolved_prediction_map[enemy_shipyard.location] 
                                        if (other_ship.halite<=ship.halite 
                                            and other_ship.probability>strategy.min_probability 
                                            and other_ship.team != state.team)])==0:
                                    
                                    if state.obs['players'][enemy_shipyard.team][0]<500:
                                        turn.next_actions[ship.ship_id] = infer_ship_action(start_point=ship.location, end_point=enemy_shipyard.location)
                                        turn.next_ship_locations.append(enemy_shipyard.location)
                                        turn.team_ships -= 1
                                        attack_probability = 1.0
                
                # look at possibility of attacking enemy ships
                for other_ship in unresolved_prediction_map[adjacent_cell]:
                    if ship.ship_id not in turn.next_actions:
                        if other_ship.team != state.team:
                            if (other_ship.halite > ship.halite and other_ship.probability > strategy.good_probability 
                                                                and other_ship.probability>attack_probability
                                                                and adjacent_cell in ship_destinations):

                                turn.next_actions[ship.ship_id] = infer_ship_action(start_point=ship.location, end_point=adjacent_cell)
                                turn.next_ship_locations.append(adjacent_cell)
                                turn.team_ships -= 1
                                attack_probability = other_ship.probability+0.1

                            elif (other_ship.halite > ship.halite and other_ship.probability > strategy.good_probability 
                                                                and other_ship.probability>attack_probability):

                                turn.next_actions[ship.ship_id] = infer_ship_action(start_point=ship.location, end_point=adjacent_cell)
                                turn.next_ship_locations.append(adjacent_cell)
                                turn.team_ships -= 1
                                attack_probability = other_ship.probability

                            elif (other_ship.halite > ship.halite and other_ship.probability > strategy.min_probability 
                                                                and other_ship.probability>attack_probability
                                                                and adjacent_cell in ship_destinations):

                                turn.next_actions[ship.ship_id] = infer_ship_action(start_point=ship.location, end_point=adjacent_cell)
                                turn.next_ship_locations.append(adjacent_cell)
                                attack_probability = other_ship.probability

            # assign next action
            if ship.ship_id not in turn.next_actions:
                if len(ship_destinations)>1:
                    selected_destination = list(ship_destinations.keys())[random.randrange(len(ship_destinations))]
                    turn.next_actions[ship.ship_id] = ship_destinations[selected_destination]
                    #turn.next_actions[ship.ship_id] = infer_ship_action(start_point=ship.location, 
                    #                                                   end_point=selected_destination)
                    turn.next_ship_locations.append(selected_destination)
                elif len(ship_destinations)==1:
                    turn.next_actions[ship.ship_id] = list(ship_destinations.values())[0]
                    turn.next_ship_locations.append(list(ship_destinations.keys())[0])
                else:
                    turn.next_actions[ship.ship_id] = None
                    turn.next_ship_locations.append(ship.location)
                    
    return turn
                    
          
def koopa_spawn(state, turn, strategy):
    # defines the decision to build a ship
    
    #spawn a ship as long as there is no ship already moved to this shipyard    
    for shipyard in state.shipyards:
        if (turn.team_halite >= 500):
            if turn.team_ships < strategy.max_ships: 
                if (shipyard.location not in turn.next_ship_locations):
                    #spawn one
                    turn.next_actions[shipyard.shipyard_id] = 'SPAWN'
                    turn.next_ship_locations.append(shipyard.location)
                    turn.team_ships += 1
                    turn.team_halite -= 500
    
    # if an enemy ship is about to attack shipyard, spawn a ship for defense
            
    return turn
#  
#  
#  -------------------------------------------  
#  4) Agent Function                            
#  -------------------------------------------  
#  

need_init = True

# Returns the commands we send to our ships and shipyards, must be last function in file
def agent(obs, config):
    global STATE
    global need_init
    
    # initialize the STATE variable for the first run
    if need_init:
        STATE = State(new_obs=obs)
        need_init = False
    
    # Define new state, strategy, and objectives
    STATE.update(new_obs=obs)
    strategy = KoopaStrategy(STATE)
    ship_objectives = optimus_objectives(team_num=STATE.team, obs=obs)
    
    # Predictions of other ship actions
    unres_map = random_prediction(STATE)
    
    # Calculate best actions for the turn
    turn = Turn(new_obs=obs)
    turn = koopa_convert(state=STATE, turn=turn, unresolved_prediction_map=unres_map, strategy=strategy)
    turn = koopa_path(ship_objectives=ship_objectives, state=STATE, turn=turn, unresolved_prediction_map=unres_map, strategy=strategy)
    turn = koopa_spawn(STATE, turn=turn, strategy=strategy)

    turn.next_actions = remove_none(turn.next_actions)
    
    return turn.next_actions
