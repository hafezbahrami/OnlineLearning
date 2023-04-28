# Ref: Algorithm1: https://arxiv.org/pdf/1209.3353.pdf
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
    rew = [1 if avg[i] > np.random.random() else 0 for i in range(len(avg))]
    return np.array(rew)


class TS():

    def __init__(self, avg):  ## Initialization
        self.means = avg
        self.num_arms = avg.size
        self.best_arm = np.argmax(self.means)
        self.restart()

    def restart(self):
        """
        Restart the algorithm: Reset the values of self.alpha and self.beta to ones (done).
        Reset the cumulative regret vector to zero.
        """
        self.alpha = np.ones(self.num_arms) # alpha = S_i ==> Successes
        self.beta = np.ones(self.num_arms)  # beta = F_i ==> Failures
        self.cum_reg = [0]
        self.sampled_mu = np.ones(self.num_arms)

    def get_best_arm(self):
        """For each time index, find the best arm according to Thompson Sampling"""
        best_arm_idx = np.argmax(self.sampled_mu)
        return best_arm_idx

    def update(self, arm_idx, rew):
        """Update the alpha and beta vectors"""
        if rew == 1:
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx] += 1

    def update_reg(self, arm_idx, rew_vec):
        """Update the cumulative regret vector"""
        increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        self.cum_reg.append(self.cum_reg[-1] + increment)

    def iterate(self, rew_vec):
        self.sampled_mu = np.array([np.random.beta(self.alpha[i], self.beta[i], size=1).item() for i in range(self.num_arms)])
        best_arm_idx = self.get_best_arm()
        binomial_reward = np.random.binomial(n=1, p=rew_vec[best_arm_idx], size=1).item()
        self.update(arm_idx=best_arm_idx, rew=binomial_reward)
        self.update_reg(arm_idx=best_arm_idx, rew_vec=rew_vec)


def run_algo(avg, num_iter, num_inst):
    reg = np.zeros((num_inst, num_iter))
    algo = TS(avg)

    for k in range(num_inst):
        algo.restart()
        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)
        for t in range(num_iter - 1):
            rew_vec = get_reward(avg)
            algo.iterate(rew_vec)
        reg[k, :] = np.asarray(algo.cum_reg)

    return reg

avg = np.asarray([0.30, 0.25, 0.20, 0.15, 0.10])
num_iter, num_inst = int(1e4), 50

reg = run_algo(avg, num_iter, num_inst)
avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg, log_x_axis=False)
zz = -1