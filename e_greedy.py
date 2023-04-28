import numpy as np
from matplotlib import pyplot as plt

def _plot(iteration_vals, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for $\epsilon$-greedy"
    colors = ["blue", "red", "green", "orange", "maroon", "deeppink", "black"]
    fig, ax = plt.subplots()
    m_vals = iteration_vals
    x = []
    y = []
    for i in range(len(m_vals)):
        x.append(i+1)
        y.append(m_vals[i])
    ax.plot(x, y, label="Cumulative Regret", color=colors[0])
    if log_x_axis:
        ax.set_xscale('log')
    ax.legend(loc='center right', fontsize='x-large')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(plot_title)
    ax.legend()


def get_reward(avg):
  l = len(avg)
  mu, sigma = 0, 1.0
  noise = np.random.normal(mu, sigma, size=l) # we could have used np.random.multivariate_normal
  return avg + noise


class egreedy():
    def __init__(self, avg, num_iter):  ## Initialization

        self.means = avg # the vector of true means of the arms
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.C = 1 # the parameter used in exercise 6.7-b in the book
        self.num_iter = num_iter
        sort = np.sort(avg)[::-1]
        self.delta = sort[0] - sort[1]

        self.restart()

    def restart(self):
        """Restart the algorithm: Reset the time index to zero and epsilon to 1 (done), the values of the
         empirical means, number of pulls, and cumulative regret to zero."""
        self.time = int(0.0)
        self.eps = 1. # probability of choosing an arm uniformly at random at time t
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.cumreg = np.zeros(self.num_iter)

    def get_best_arm(self):
        """For each time index, find the best arm according to e-greedy"""
        return np.argmax(self.emp_means).item()


    def update_stats(self, rew, arm_idx):
        """Update the empirical means, the number of pulls, epsilon, and increment the time index"""
        self.time += 1
        self.num_pulls[arm_idx] += 1
        self.emp_means[arm_idx] += (rew[arm_idx] - self.emp_means[arm_idx]) / self.num_pulls[arm_idx]
        self.eps = np.min([1., (self.C*self.num_arms)/(self.delta**2*self.time)])

    def update_reg(self, rew_vec, arm_idx):
        """Update the cumulative regret"""
        Method = "What TA noted"
        if Method == "What TA noted":
            increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        else:
            increment = (self.means[self.best_arm] - self.means[arm_idx])
        self.cumreg[self.time] = self.cumreg[self.time-1] + increment

    def iterate(self, rew_vec):
        """Iterate the algorithm"""
        arm_idx = self.get_best_arm()

        explore = np.random.binomial(n=1, p=self.eps, size=1).item() # getting 1 (as success) with prob = epsilon
        if explore:
            arm_idx = np.random.randint(low=0, high=self.num_arms-1, size=1, dtype=int) # uniform normal samlping from all arms

        self.update_stats(rew_vec, arm_idx)
        self.update_reg(rew_vec, arm_idx)


def run_algo(avg, num_iter, num_inst):
    reg = np.zeros((num_inst, num_iter))
    algo = egreedy(avg, num_iter)

    for j in range(num_inst):
        algo.restart()
        if (j + 1) % 10 == 0:
            print('Instance number = ', j + 1)
        for t in range(num_iter - 1):
            rew_vec = get_reward(avg)
            algo.iterate(rew_vec)
        reg[j, :] = np.asarray(algo.cumreg)
    return reg

import math
avg = np.asarray([0.96, 0.7, 0.5, 0.6, 0.1])
num_iter, num_inst = int(1*1e4), 10

reg = run_algo(avg, num_iter, num_inst)

avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg)
zz = -1