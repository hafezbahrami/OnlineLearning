import numpy as np
from matplotlib import pyplot as plt

def _plot(iteration_vals, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for UCB" # $\epsilon$-greedy"
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
    sigma = 1.0
    cov = sigma * np.identity(avg.size)
    rew = avg + np.random.multivariate_normal(np.zeros(avg.size), cov)
    return rew


class UCB():
    def __init__(self, avg): ## Initialization
        self.means = avg
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.restart()

    def restart(self):
        """Restart the algorithm: Reset the time index to zero and the upper confidence values to high
        values (done).
        Set the values of the empirical means, number of pulls, and cumulative regret vector to zero.
        """
        self.time = int(0.0)
        self.ucb_arr = 1e5 * np.ones(self.num_arms)
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.cum_reg = [0]

    def get_best_arm(self):
        """For each time index, find the best arm according to UCB"""
        # play each arm once, before we start the UCN algo based on confidence index
        if self.time < self.num_arms:
            return self.time
        self.update_ucb()
        return np.argmax(self.ucb_arr).item()

    def update_stats(self, rew, arm_idx):
        """Update the empirical means, the number of pulls, and increment the time index"""
        self.time += 1
        self.num_pulls[arm_idx] += 1
        self.emp_means[arm_idx] += (rew[arm_idx] - self.emp_means[arm_idx]) / self.num_pulls[arm_idx]

    def update_ucb(self):
        """Update the vector of upper confidence bounds"""
        ft = 1.0 + (self.time+1) * (np.log(self.time+1))**2
        self.ucb_arr = self.emp_means + np.sqrt(2.0*np.log(ft)/self.num_pulls)

    def update_reg(self, rew_vec, arm_idx):
        """Update the cumulative regret"""
        increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        self.cum_reg.append(self.cum_reg[-1] + increment)

    def iterate(self, rew_vec):
        """Iterate the algorithm. A is active arm list"""
        best_arm_idx = self.get_best_arm()
        self.update_stats(rew_vec, best_arm_idx)
        self.update_reg(rew_vec, best_arm_idx)


def run_algo(avg, num_iter, num_inst):
    reg = np.zeros((num_inst, num_iter))
    algo = UCB(avg)

    for k in range(num_inst):
        algo.restart()
        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)
        for t in range(num_iter - 1):
            rew_vec = get_reward(avg)
            algo.iterate(rew_vec)
        reg[k, :] = np.asarray(algo.cum_reg)
    return reg


avg = np.asarray([0.8, 0.96, 0.7, 0.5, 0.4, 0.3])
num_iter, num_inst = int(5e4), 30

reg = run_algo(avg, num_iter, num_inst)

avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg, log_x_axis=True)
zz = -1