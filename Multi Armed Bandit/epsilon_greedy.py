import numpy as np
import matplotlib.pyplot as plt
# THIS PROGRAM SOLVES MULTI ARMED PROBLEM USING EPSILON GREEDY ALGORITHM

# DEFINE AND CHARACTERISE A BANDIT

class Bandit:
    def __init__(self,m):
        self.N=0 # NUMBER OF TIMES THIS ARM HAS BEEN PULLED
        self.m=m # MEAN VALUE REWARD BY PULLING THIS ARM
        self.mean=0 # MEAN REWARD ACCUMULATED BY PULLING THIS ARM
    def pull(self):
        return np.random.randn()+self.m
    def update(self,x):
        self.N+=1
        self.mean=(1-1/self.N)*self.mean + 1/self.N*x

def run_epsilon_greedy(means,epsilon,num_iterations):
    bandits=[]
    for m in means:
        bandits.append(Bandit(m))
    rewards=np.empty(num_iterations)
    for i in range(num_iterations):
        p=np.random.random()
        if p<epsilon:
            arm=np.random.choice(len(means))
        else:
            arm=np.argmax([b.mean for b in bandits])
        x=bandits[arm].pull()
        bandits[arm].update(x)
        rewards[i]=x
    cum_avg_rewards=np.cumsum(rewards)/(np.arange(num_iterations)+1)
    # plt.plot(cum_avg_rewards)
    # plt.xscale('log')
    # plt.show()
    print("Mean of bandits are as follows:")
    for b in bandits:
        print(b.mean)
    return cum_avg_rewards

if __name__=='__main__':
    c_1 = run_epsilon_greedy([1.0, 2.0, 3.0], 0.1, 100000)
    c_05 = run_epsilon_greedy([1.0, 2.0, 3.0], 0.05, 100000)
    c_01 = run_epsilon_greedy([1.0, 2.0, 3.0], 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()


    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()
