import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_epsilon_greedy

class Bandit():
    def __init__(self,actual_mean,optimistic_value):
        self.actual_mean=actual_mean
        self.N=1
        self.mean=optimistic_value
    def pull(self):
        return np.random.randn()+self.actual_mean
    def update(self,x):
        self.N+=1
        self.mean=(1-1.0/self.N)*self.mean+1/self.N*x

def run_oiv(means,optimistic_value,num_iterations):
    rewards=np.empty(num_iterations)
    bandits=[Bandit(m,optimistic_value) for m in means]
    for i in range(num_iterations):
        arm=np.argmax([b.mean for b in bandits])
        reward=bandits[arm].pull()
        bandits[arm].update(reward)
        rewards[i]=reward
    cum_avg_rewards=np.cumsum(rewards)/(np.arange(num_iterations)+1)
    # plt.plot(cum_avg_rewards)
    # plt.xscale('log')
    # plt.show()
    return cum_avg_rewards

if __name__=='__main__':
    eps = run_epsilon_greedy([1.0, 2.0, 3.0], 0.1, 100000)
    oiv = run_oiv([1.0, 2.0, 3.0],10 ,100000)

    plt.plot(eps,label='epsilon=0.1')
    plt.plot(oiv,label='optimistic_value')
    plt.legend()
    # plt.xscale('log')
    plt.show()
