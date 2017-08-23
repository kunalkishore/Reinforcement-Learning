import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_epsilon_greedy
from optimistic_intial_values import run_oiv
import math

class Bandit():
    def __init__(self,actual_mean):
        self.actual_mean=actual_mean
        self.N=0
        self.mean=0
    def pull(self):
        return np.random.randn()+self.actual_mean
    def update(self,x):
        self.N+=1
        self.mean=(1-1/self.N)*self.mean+1/self.N*x

def ucb(arm_mean,total_pulls,arm_pulls):
    if arm_pulls==0:
        return float('inf')
    return arm_mean + math.sqrt(2*np.log(total_pulls)/arm_pulls)

def run_ucb1(means,num_iterations):
    bandits = [Bandit(m) for m in means]
    rewards=np.empty(num_iterations)
    for i in range(num_iterations):
        arm=np.argmax([ucb(b.mean,i,b.N) for b in bandits ])
        reward=bandits[arm].pull()
        bandits[arm].update(reward)
        rewards[i]=reward
    cum_avg_rewards=np.cumsum(rewards)/(np.arange(num_iterations)+1)
    # plt.plot(cum_avg_rewards)
    # plt.xscale('log')
    # plt.show()
    return cum_avg_rewards

if __name__=='__main__':
    eps = run_epsilon_greedy([1.0,2.0,3.0],0.1,100000)
    ucb1 = run_ucb1([1.0,2.0,3.0],100000)
    oiv = run_oiv([1.0, 2.0, 3.0],10 ,100000)
    plt.plot(eps,label='epsilon=0.1')
    plt.plot(ucb1,label='UCB-1')
    plt.plot(oiv,label='optimistic_value')
    plt.legend()
    plt.xscale('log')
    plt.show()
