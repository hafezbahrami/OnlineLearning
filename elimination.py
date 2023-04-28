import numpy as np
from matplotlib import pyplot as plt

def _plot(iteration_vals, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for Elimination" # $\epsilon$-greedy"
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
    sigma=0.01
    cov = sigma * np.identity(avg.size)
    rew = avg + np.random.multivariate_normal(np.zeros(avg.size), cov)
    return rew


class Elimination():
    def __init__(self, avg, num_iter ): ## Initialization
        self.means = avg
        self.num_iter = num_iter
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.time = int(0.0)
        self.A = np.arange(self.num_arms)
        self.cum_reg = [0]
        self.m = np.ceil( 2 ** (2 * self.time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.time))))
        self.restart()

    def restart(self):
        """Restart the algorithm: Reset the time index to zero and epsilon to 1 (done), the values of the
         empirical means, number of pulls, and cumulative regret to zero."""
        self.time2 = int(0.0)
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)

    def get_best_arm(self):
        """For each time index, find the best arm according to e-greedy"""
        return np.argmax(self.emp_means).item()

    def update_stats(self, rew, arm_idx):
        """Update the empirical means, the number of pulls, epsilon, and increment the time index"""
        self.time2 += 1
        self.num_pulls[arm_idx] += 1
        self.emp_means[arm_idx] += (rew[arm_idx] - self.emp_means[arm_idx]) / self.num_pulls[arm_idx]

    def update_elim(self):  ## Update the active set
        self.m = np.ceil(
            2 ** (2 * self.time) * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.time))))
        updated_active_arm_idx = []
        max_emp_mean = np.max(self.emp_means)
        for active_arm_idx in self.A:
            if self.emp_means[active_arm_idx] + 2**(-self.time) >= max_emp_mean:
                updated_active_arm_idx.append(active_arm_idx)
        self.A = np.array(updated_active_arm_idx)

    def update_reg(self, rew_vec, arm_idx):
        """Update the cumulative regret"""
        increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        self.cum_reg.append(self.cum_reg[-1] + increment)

    def iterate(self, rew_vec):
        """Iterate the algorithm. A is active arm list"""
        idx = int(self.time2 % int(self.A.size))
        self.update_stats(rew_vec, self.A[idx])
        self.update_reg(rew_vec, self.A[idx])


def run_algo(avg, num_iter, num_inst):
    reg = np.zeros((num_inst, num_iter))

    for k in range(num_inst):
        algo = Elimination(avg, num_iter)
        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)
        while len(algo.cum_reg) <= num_iter:
            if len(algo.cum_reg) >= num_iter:
                break
            for t in range(int(algo.m) * algo.A.size):
                if len(algo.cum_reg) >= num_iter:
                    break
                else:
                    rew_vec = get_reward(avg)
                    algo.iterate(rew_vec)
            algo.update_elim()
            algo.restart()
            algo.time += 1
        reg[k, :] = np.asarray(algo.cum_reg)
    return reg


avg = np.asarray([0.8, 0.88, 0.5, 0.7, 0.65])
num_iter, num_inst = int(1500), 250

reg = run_algo(avg, num_iter, num_inst)

avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg, log_x_axis=True)
zz = -1