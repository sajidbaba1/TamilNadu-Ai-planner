import sys
import numpy as np

sys.path.insert(0, '.')

# Simulate the exact shares_wall check on this known-good placement
WALL_TOL = 0.35
OVERLAP_TOL = 0.10


def shares_wall(a, b):
    y_touch = (abs(a['y'] + a['d'] - b['y']) < WALL_TOL or
               abs(b['y'] + b['d'] - a['y']) < WALL_TOL)
    x_overlap = (a['x'] < b['x'] + b['w'] - WALL_TOL and
                 a['x'] + a['w'] > b['x'] + WALL_TOL)
    x_touch = (abs(a['x'] + a['w'] - b['x']) < WALL_TOL or
               abs(b['x'] + b['w'] - a['x']) < WALL_TOL)
    y_overlap = (a['y'] < b['y'] + b['d'] - WALL_TOL and
                 a['y'] + a['d'] > b['y'] + WALL_TOL)
    return (y_touch and x_overlap) or (x_touch and y_overlap)


living = {'x': 0.0, 'y': 6.9, 'w': 3.07, 'd': 3.43}
verandah = {'x': 0.0, 'y': 10.33, 'w': 10.0, 'd': 1.17}
dining = {'x': 3.07, 'y': 6.9, 'w': 2.37, 'd': 2.5}
master = {'x': 0.0, 'y': 0.0, 'w': 2.93, 'd': 3.24}
toilet_a = {'x': 0.0, 'y': 3.24, 'w': 1.2, 'd': 1.95}

print('living top:    ', round(living["y"] + living["d"], 3))
print('verandah bot:  ', verandah["y"])
print('y_diff:        ', round(abs(living["y"] + living["d"] - verandah["y"]), 4))
print('WALL_TOL:      ', WALL_TOL)
print()

y_touch = abs(living['y'] + living['d'] - verandah['y']) < WALL_TOL
x_overlap = (living['x'] < verandah['x'] + verandah['w'] - WALL_TOL and
             living['x'] + living['w'] > verandah['x'] + WALL_TOL)
print(f'living-verandah y_touch:   {y_touch}  (diff={abs(living["y"]+living["d"]-verandah["y"]):.4f})')
print(f'living-verandah x_overlap: {x_overlap}')
print(f'shares_wall result:        {shares_wall(living, verandah)}')
print()
print(f'master-toilet y_diff:      {abs(master["y"]+master["d"]-toilet_a["y"]):.4f}')
print(f'master-toilet shares_wall: {shares_wall(master, toilet_a)}')
