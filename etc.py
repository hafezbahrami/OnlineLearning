import numpy as np
from matplotlib import pyplot as plt

def _plot(iteration_vals, vertical_line_x, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for $ETC$"
    colors = ["blue", "red", "green", "orange", "maroon", "deeppink", "black"]
    fig, ax = plt.subplots()
    m_vals = iteration_vals
    x = []
    y = []
    for i in range(len(m_vals)):
        x.append(i+1)
        y.append(m_vals[i])
    ax.plot(x, y, label="Cumulative Regret", color=colors[0])
    ax.axvline(x=vertical_line_x, color='b', label='axvline - full height') # adding a vertical line
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


class ETC():
    def __init__(self, avg, m, regret_method):  ## Initialization

        self.regret_method = regret_method
        self.means = avg
        self.m = m
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.num_iter = num_iter # time horizen
        self.restart()

    def restart(self):
        """
        Restart the algorithm: Reset the time index to zero (done), the values of the empirical means,
        number of pulls, and cumulative regret to zero.
        """
        self.time = int(0)
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.cumreg = np.zeros(self.num_iter)


    def get_best_arm(self):
        """For each time index, find the best arm according to ETC."""
        return np.argmax(self.emp_means)

    def update_stats(self, rew, arm_idx):
        """Update the empirical means, the number of pulls, and increment the time index"""
        self.time += 1
        self.num_pulls[arm_idx] += 1
        self.emp_means[arm_idx] += (rew[arm_idx] - self.emp_means[arm_idx]) / self.num_pulls[arm_idx]

    def update_reg(self, rew_vec, arm_idx):
        """Update the cumulative regret"""
        if self.regret_method == "based_on_rew_vector":
            increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        else:
            increment = (self.means[self.best_arm] - self.means[arm_idx])
        self.cumreg[self.time] = self.cumreg[self.time-1] + increment



    def iterate(self, rew_vec):
        """Iterate the algorithm"""
        # Pure Exploration Phase
        if self.time < int(self.m * self.num_arms):
            for arm_idx in range(self.num_arms):
                self.update_stats(rew_vec, arm_idx)
                self.update_reg(rew_vec, arm_idx)
        # Pure Exploitation Phase
        else:
            selected_best_arm_idx = np.argmax(self.emp_means)
            self.update_stats(rew_vec, selected_best_arm_idx)
            self.update_reg(rew_vec, selected_best_arm_idx)


def run_algo(avg, m, num_iter, num_inst, regret_method):
    reg = np.zeros((num_inst, num_iter))
    algo = ETC(avg, m, num_iter, regret_method)

    for j in range(num_inst):
        algo.restart()
        if (j + 1) % 10 == 0:
            print('Instance number = ', j + 1)
        for t in range(num_iter - 1):
            if algo.time < num_iter-1:
                rew_vec = get_reward(avg)
                algo.iterate(rew_vec)

        reg[j, :] = np.asarray(algo.cumreg)
    return reg

import math
avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
num_iter, num_inst = int(1e4), 30

Delta_1 = np.sort(-(avg - np.max(avg)))[1]
m_temp = 4/Delta_1**2 * np.log(num_iter*Delta_1**2/4)
m = np.max([1,  int(m_temp)]) ## Your code to update m here. (Hint: Use equation (6.5) in the book)

if (m*len(avg)) > num_iter:
    num_iter = (m*len(avg)) + 1000

regret_method = "based_on_rew_vector"
# regret_method = "based_on_means_vector"
reg = run_algo(avg, m, num_iter, num_inst, regret_method)

avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg, vertical_line_x=m*len(avg))

zz = -1