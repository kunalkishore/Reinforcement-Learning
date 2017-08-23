import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_epsilon_greedy
from optimistic_intial_values import run_oiv
from ucb1 import run_ucb1

class BayesianBandit():
    def __init__(self,actual_mean):
        self.actual_mean=actual_mean
        self.predicted_mean=0
        self.tau = 1
        self.lambda_ = 1
        self.rewards=0
        self.N=1
    def pull(self):
        return np.random.randn()+self.actual_mean
    def sample(self):
        return np.random.randn()/np.sqrt(self.lambda_)+self.predicted_mean
    def update(self,x):
        self.N+=1
        lambda_new = self.lambda_+ self.tau*self.N
        self.rewards+=x
        self.predicted_mean = ((self.predicted_mean)*self.lambda_ + (self.tau*self.rewards))/(lambda_new)
        self.lambda_=lambda_new

def run_thompson_sampling(means,num_iterations):
    bandits = [BayesianBandit(m) for m in means]
    rewards=np.empty(num_iterations)
    for i in range(num_iterations):
        arm  = np.argmax([b.sample() for b in bandits])
        reward=bandits[arm].pull()
        bandits[arm].update(reward)
        rewards[i]=reward
    cum_avg_rewards=np.cumsum(rewards)/(np.arange(num_iterations)+1)
    plt.plot(cum_avg_rewards)
    plt.xscale('log')
    plt.show()
    return cum_avg_rewards

if __name__ == '__main__':
    eps=run_epsilon_greedy([1.0,2.0,3.0],0.1,100000)
    oiv=run_oiv([1.0,2.0,3.0],10,100000)
    ucb=run_ucb1([1.0,2.0,3.0],100000)
    thompson =run_thompson_sampling([1.0,2.0,3.0],100000)
    plt.plot(eps,label='epsilon-greedy')
    plt.plot(oiv,label='optimistic-value')
    plt.plot(ucb,label='ucb1')
    plt.plot(thompson,label='thompson')
    plt.legend()
    plt.xscale('log')
    plt.show()
