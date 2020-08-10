from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from dataclasses import dataclass
import numpy as np
from itertools import product
import time

# action names
NORTH = 'N'
SOUTH = 'S'
EAST = 'E'
WEST = 'W'
MINE = 'M'

@dataclass
class Action:
    to_point: Point
    action: str

# actions and where they come from
possible_actions = [
    Action(Point(0, 0), MINE),
    Action(Point(1, 0), WEST),
    Action(Point(-1, 0), EAST), 
    # North and south are backward
    Action(Point(0, 1), SOUTH), 
    Action(Point(0, -1), NORTH)
]

@dataclass
class Route:
    # path is a str of concatenated action names
    path: str
    # expected amount of halite gained from this route
    value: int

# where an action would take you
dir_lookup = {
    EAST: Point(1, 0),
    WEST: Point(-1, 0),
    # north and south are backward
    NORTH: Point(0,1),
    SOUTH: Point(0,-1),
    MINE: Point(0, 0)
}

# Apparently Points are expensive to use
dir_lookup_x = {
    EAST: 1,
    WEST: -1,
    NORTH: 0,
    SOUTH: 0,
    MINE: 0
}

dir_lookup_y = {
    EAST: 0,
    WEST: 0,
    NORTH: 1,
    SOUTH: -1,
    MINE: 0
}

def score_path(path: str, start_point_x: int, start_point_y: int, halite: np.array, prev_score: float, board_size=21):
    ''' find the expected score from path
    path is a str of concatenated action names
    prev_score is the score for path[1:]
    '''
    curr_pos_x = 0
    curr_pos_y = 0
    if path[0] != MINE:
        return prev_score
    point_halite = halite[start_point_x, start_point_y]
    for p in path[1:]:
        if p == MINE and curr_pos_x == 0 and curr_pos_y == 0:
            point_halite *= 0.75
        curr_pos_x = (curr_pos_x + dir_lookup_x[p]) % board_size
        curr_pos_y =  (curr_pos_y + dir_lookup_y[p]) % board_size    
    score = point_halite * 0.25 + prev_score
    return score

def find_best_paths(board, board_size, t_max, targets, time_max):
    halite = np.zeros((board_size, board_size))
    for point, cell in board.cells.items():
        halite[point.x, point.y] = cell.halite
    # m is a matrix of starting point to best route to a dest
    # (or alternatively dest point to best route to a start)
    m = np.full((board_size, board_size), None)
    # set destinations as empty solution
    for target in targets:
        m[target.x, target.y] = Route('', 0)
    # these are the next values for m
    m_next = np.full((board_size, board_size), None)
    for t in range(1, t_max + 1):
        for pos in product(range(board_size), repeat=2):
            if time.time() > time_max:
                return m
            # if there was a path to dest from here @ t-1...
            if m[pos] is not None:
                # then there must be a viable path to all of my neighbors @ t
                for possible in possible_actions:
                    # p is the neighbor of pos
                    px = (pos[0] + possible.to_point.x) % board_size
                    py = (pos[1] + possible.to_point.y) % board_size
                    # the path that adds p to the best route of pos
                    ppath = possible.action + m[pos].path
                    # The expected score if we follow this route
                    p_score = score_path(ppath, px, py, halite, m[pos].value)
                    # we have found the first path for p or we have found a better path for p
                    if m_next[px, py] is None or p_score > m_next[px, py].value:
                        m_next[px, py] = Route(ppath, p_score)
        #for y in range(board_size):
        #    print([r.value if r is not None else ' '*t for r in m_next[:,y]])
        m = m_next
        m_next = np.full((board_size, board_size), None)
    return m

action_lookup = {
    'N': ShipAction.NORTH,
    'S': ShipAction.SOUTH,
    'E': ShipAction.EAST,
    'W': ShipAction.WEST,
    'M': None
}

def mining_locs(path: str, start_point: Point, board_size=21):
    curr_pos = start_point
    mining_locs = []
    for p in path:
        if p == MINE:
            mining_locs.append(curr_pos)
            continue
        curr_pos = (dir_lookup[p] + curr_pos) % board_size
    return mining_locs

def path_to_action_list(path: str):
    return [action_lookup[p] for p in path]