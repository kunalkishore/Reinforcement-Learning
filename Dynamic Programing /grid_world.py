import numpy as np


class Grid():
    def __init__(self,height,width,state):
        self.height=height
        self.width = width
        self.i = state[0]
        self.j = state[1]

    def set_actions_and_rewards(self,actions, rewards):
        '''Assuming actions consists of states other than terminal states and walls.
        Rewards consists of only terminal states and in other states we assume reward to be 0'''
        self.actions=actions
        self.rewards = rewards

    def set_current_state(self,state):
        self.i=state[0]
        self.j=state[1]

    def get_state(self):
        return (self.i,self.j)

    def is_terminal(self,state):
        return state not in self.actions

    def move(self,action):
        if action in self.actions[(self.i,self.j)]:
            if action=='U':
                self.i-=1;
            elif action=='D':
                self.i+=1
            elif action=='L':
                self.j-=1
            elif action == 'R':
                self.j+=1
        return self.rewards.get((self.i,self.j),0)

    def game_over(self):
        return (self.i,self.j) not in self.actions

    def all_states(self):
        return set(list(self.rewards.keys())+list(self.actions.keys()))

    def draw_grid(self):
        for i in range(self.height):
            print("________________________")
            s=""
            for j in range(self.width):
                if (i,j) in self.actions:
                    reward=0
                if (i,j) in self.rewards:
                    reward=self.rewards[(i,j)]
                else:
                    reward=' /  '
                s+=str(reward)+"  |"
            print(s)
        print("________________________")

def make_grid():
    '''
    |__|__|__|+1|
    |__|__|//|__|
    |__|//|__|-1|
    |S |__|__|__|
    '''
    grid = Grid(4,4,(3,0))
    rewards = {
    (0,3):+1,
    (2,3):-1,
    (0,0):-0.1,
    (0,1):-0.1,
    (0,2):-0.1,
    (1,0):-0.1,
    (1,1):-0.1,
    (1,3):-0.1,
    (2,0):-0.1,
    (2,2):-0.1,
    (3,0):-0.1,
    (3,1):-0.1,
    (3,2):-0.1,
    (3,3):-0.1
    }
    actions = {
    (0,0):['R','D'],
    (0,1):['R','L','D'],
    (0,2):['R','L'],
    (1,0):['R','U','D'],
    (1,1):['U','L'],
    (1,3):['U','D'],
    (2,0):['U','D'],
    (2,2):['R','D'],
    (3,0):['U','R'],
    (3,1):['L','R'],
    (3,2):['U','L','R'],
    (3,3):['U','L']
    }
    grid.set_actions_and_rewards(actions,rewards)
    grid.draw_grid()
    return grid


if __name__=='__main__':
    make_grid()
