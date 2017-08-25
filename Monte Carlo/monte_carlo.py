import numpy as np
from grid_world import make_grid
import matplotlib.pyplot as plt

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

def random_action(action,epsilon=0.1):
    p = np.random.random()
    if p < epsilon:
        return np.random.choice(ALL_ACTIONS)
    else:
        return action

def max_dict(action_reward):
    key=None
    value = float('-inf')
    for action,reward  in action_reward.items():
        if reward>value:
            value=reward
            key = action
    return [key,value]

def play_game(policy,grid):
    state=(3,3)
    grid.set_current_state(state)
    action=random_action(policy[state])
    state_action_reward=[(state,action,0)]
    while True:
        reward = grid.move(action)
        state = grid.get_state()
        if grid.game_over():
            state_action_reward.append((state,None,reward))
            break
        else:
            action = random_action(policy[state])
            state_action_reward.append((state,action,reward))

    G=0
    state_action_returns=[]
    first=True
    for s,a,r in reversed(state_action_reward):
        if first:
            first=False
        else:
            state_action_returns.append((s,a,G))
        G=r+GAMMA*G
        state_action_returns.reverse()
    return state_action_returns


if __name__ == '__main__':
    grid = make_grid()
    all_states = list(grid.actions.keys())
    Q={}
    returns={}
    for state in all_states:
        Q[state]={}
        for action in grid.actions[state]:
            Q[state][action]=0
            returns[(state,action)]=[]
    policy={}
    for state in grid.actions.keys():
        policy[state]=np.random.choice(grid.actions[state])
    delta=[]
    for i in range(5000):
        state_action_reward = play_game(policy,grid)
        seen=set()
        biggest_change=0
        for s,a,r in state_action_reward:
            if (s,a) not in seen and a in grid.actions[s]:
                seen.add((s,a))
                old_q=Q[s][a]
                returns[(s,a)].append(r)
                Q[s][a]=(np.mean(returns[(s,a)]))
                biggest_change=max(biggest_change,np.abs(Q[s][a]-old_q))
        delta.append(biggest_change)

        for s in grid.actions.keys():
            a,_ = max_dict(Q[s])
            policy[s]=a

    print("final policy:")
    print_policy(policy, grid)

    value={}
    for s in policy.keys():
        value[s]=Q[s][policy[s]]
    print("Final Values:")
    print_values(value,grid)
    plt.plot(delta)
    plt.show()
