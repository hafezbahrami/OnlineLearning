import numpy as np
from matplotlib import pyplot as plt

def _plot(iteration_vals, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for Aversarial" # $\epsilon$-greedy"
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


def get_reward(avg, var):
    cov = var * np.identity(avg.size)
    mean = np.zeros(avg.size)
    rew = avg + np.random.multivariate_normal(mean=mean, cov=cov)
    return rew


class EXP3:
    def __init__(self, avg, eta): ## Initialization
        self.means = avg
        self.num_arms = avg.size
        self.eta = eta
        self.best_arm = np.argmax(avg)
        self.restart()

    def restart(self):
        """
        Reset self.time to zero
        Set the values of the num_plays, S, and cum_reg to zero,
        and set probs_arr to be uniform
        """
        self.time = int(0.0)
        self.num_pulls = np.zeros(self.num_arms) # = self.num_plays[arm]: the vector of number of times that arm k has been pulled
        self.cum_reg = [0]
        self.S = np.zeros(self.num_arms) # vector of estimated reward by the end of time t
        self.probs_arr = np.array([1.]*self.num_arms) / self.num_arms # P_th: uniform normal samlping from all arms # sampling distribution vector P_t

    def get_best_arm(self):
        """
            For each time index, find the best arm according to EXP3
            Hint: use np.random.choice
        :return:
        """
        arm_idx = np.random.choice(a=range(self.num_arms), size=1, p=self.probs_arr).item()
        return arm_idx

    def update_exp3(self, arm_idx, rew_vec):
        """
        Compute probs_arr and update the total estimated reward for each arm.
        Steps 4, 5, 6 in the algorithm, page 152, text book
        :param arm_idx:
        :param rew_vec:
        :return:
        """
        sum_exp_weights = 0
        for i in range(self.num_arms):
            sum_exp_weights += np.exp(self.eta*self.S[i])

        for i in range(self.num_arms):
            self.probs_arr[i] = np.exp(self.eta*self.S[i]) / sum_exp_weights

        X = rew_vec[arm_idx]

        for i in range(self.num_arms):
            indicator = 1. if i == arm_idx else 0.
            self.S[i] += (1 - (indicator*(1.-X)/self.probs_arr[i]))

    def update_reg(self, arm_idx, rew_vec):
        """
        Though the bound for the regret in adversarial setting of bandit uses "fixed arm in hindsight, but
        for the calculation we still use the procedure we had in other algorithms of stochiatstics approaches
        Update the cumulative regret
        """
        increment = rew_vec[self.best_arm] - rew_vec[arm_idx]
        self.cum_reg.append(self.cum_reg[-1] + increment)

    def iterate(self, rew_vec):
        """Iterate the algorithm. A is active arm list"""
        self.time += 1
        selected_arm_idx = self.get_best_arm()
        self.update_exp3(arm_idx=selected_arm_idx, rew_vec=rew_vec)
        self.update_reg(arm_idx=selected_arm_idx, rew_vec=rew_vec)


def run_algo(avg, eta, num_iter, num_inst, var):
    reg = np.zeros((num_inst, num_iter))
    algo = EXP3(avg, eta)

    for k in range(num_inst):
        algo.restart()
        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)
        for t in range(num_iter - 1):
            rew_vec = get_reward(avg, var)
            algo.iterate(rew_vec)
        reg[k, :] = np.asarray(algo.cum_reg)

    return reg


avg = np.asarray([0.8, 0.7, 0.5])
num_iter, num_inst = int(2e3), 20
eta = np.sqrt(np.log(avg.size) / (num_iter * avg.size))
var = 0.01

reg = run_algo(avg, eta, num_iter, num_inst, var)

avg_reg = np.mean(reg, axis=0)

# plt.plot(avg_reg)
_plot(avg_reg, log_x_axis=True)
zz = -1