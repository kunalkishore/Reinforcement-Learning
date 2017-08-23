import numpy as np
from grid_world import make_grid

EPSILON = 10e-4
GAMMA = 0.9
ALL_ACTIONS = ['U','D','R','L']

def print_policy(P, g):
  for i in xrange(g.width):
    print "---------------------------"
    for j in xrange(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a,)
    print("")

def print_values(V, g):
  for i in xrange(g.width):
    print "---------------------------"
    for j in xrange(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v,)
      else:
        print("%.2f|" % v,) # -ve sign takes up an extra space
    print("")

if __name__ == '__main__':
    policy={}
    value_function={}

    grid = make_grid()
    for state in grid.actions.keys():
        policy[state] = np.random.choice(ALL_ACTIONS)

    print("Initial Policy")
    print_policy(policy,grid)

    for state in grid.all_states():
        if state in grid.actions.keys():
            value_function[state]=np.random.random()
        else:
            value_function[state] = 0

    print("Initial Values")
    print_values(value_function,grid)

    while true:
        #POLICY EVALUATION AND VALUE UPDATE
        while true:
            biggest_change=0
            for state in grid.all_states():
                old_val = value_function[state]
                grid.set_current_state(state)
                reward = grid.move(policy[state])
                
