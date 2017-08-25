import numpy as np
from grid_world import make_grid
TOLERANCE=10e-4
ALL_ACTIONS=['U','D','L','R']
GAMMA = 0.9

def print_policy(P, g):
    for i in range(g.height):
        print("________________________")
        s=""
        for j in range(g.width):
            a = P.get((i,j),' ')
            s+=str(a)+"  |"
        print(s)
    print("________________________")

def print_values(V, g):
    for i in range(g.height):
        print("________________________")
        s=""
        for j in range(g.width):
            v = V.get((i,j), 0)
            s+=str("%.2f|" % v)+"  |"
        print(s)
    print("________________________")


if __name__ == '__main__':
    policy={}
    value={}
    grid = make_grid()
    states=grid.all_states()
    for state in states: 
        if state in grid.actions:
            value[state]=np.random.random()
        else:
            value[state]=0
    print("Initial Values")
    print_values(value,grid)
    has_converged=True
    while True:
        biggest_change=0
        for state in states:
            old_val=value[state]
            if state in grid.actions:
                max_val=float('-inf')
                for action in ALL_ACTIONS:
                    grid.set_current_state(state)
                    reward=grid.move(action)
                    new_value = reward+GAMMA*value[grid.get_state()]
                    max_val = max(max_val,new_value)
                value[state]=max_val
                biggest_change=max(biggest_change,np.abs(max_val-old_val))
        if biggest_change<TOLERANCE:
            break

    for state in grid.actions.keys():
        max_val=float('-inf')
        best_action=None
        for action in ALL_ACTIONS:
            grid.set_current_state(state)
            reward=grid.move(action)
            v = reward + GAMMA*value[grid.get_state()]
            if v>max_val:
                max_val=v
                best_action=action
        policy[state]=best_action

    print("values:")
    print_values(value, grid)
    print("policy:")
    print_policy(policy, grid)
