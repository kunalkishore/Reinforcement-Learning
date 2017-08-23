import numpy as np
from grid_world import make_grid

EPSILON = 10e-4
GAMMA = 0.9
ALL_ACTIONS = ['U','D','R','L']

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

    while True:
        #POLICY EVALUATION AND VALUE UPDATE
        while True:
            biggest_change=0
            for state in grid.all_states():
                old_val = value_function[state]
                if state in policy:
                    grid.set_current_state(state)
                    reward = grid.move(policy[state])
                    new_val = reward + GAMMA*value_function[grid.get_state()]
                    value_function[state]=new_val;
                    biggest_change = max(biggest_change,np.abs(new_val-old_val))
            if biggest_change<EPSILON:
                break

        # CONTROL - POLICY IMPROVEMENT
        policy_converged = True
        for state in grid.all_states():
            if state in policy:
                old_policy=policy[state]
                new_policy=None
                best_value=float('-inf')
                for action in ALL_ACTIONS:
                    grid.set_current_state(state)
                    reward=grid.move(action)
                    value = reward+GAMMA*value_function[grid.get_state()]
                    if value>best_value:
                        best_value=value
                        new_policy=action
                policy[state]=new_policy
                if old_policy!=new_policy:
                    policy_converged=False

        if policy_converged:
            break
    print("Final Value Function")
    print_values(value_function,grid)
    print("Final Policy")
    print_policy(policy,grid)
