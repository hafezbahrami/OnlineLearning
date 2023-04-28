import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm

available_arms = np.array([
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1)])


def _plot(iteration_vals, log_x_axis=False):
    x_label = "time"
    y_label = "Cum-Regret" #"$log( \| x^{(k)} - x^* \|_{2}^{2} )$"
    plot_title = "Cumulative Regret for Linear UCB" # $\epsilon$-greedy"
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


class context_arm(object):
    def __init__(self, available_arms=available_arms):
        self.theta = np.array((0.1, 0.4, 0.2, 0.3))
        self.available_arms = np.array(available_arms)
        self.noise_var = 0.25
        self.mean = 0.0

    def pull_arm(self, arm_idx):
        """Return X_t given the index of the arm played"""
        noise = np.random.normal(loc=self.mean, scale=self.noise_var, size=None)
        reward = np.inner(self.theta, self.available_arms[arm_idx])
        return reward + noise

    def genie_reward(self):
        """Return the genie reward"""
        reward_vec = np.inner(self.theta, self.available_arms)
        return np.max(reward_vec)


class LinUCB():

    def __init__(self, available_arms):  # Initialization

        self.arms = available_arms
        self.num_arms = len(self.arms)
        self.d = len(self.arms[0])
        self.reward_history = []
        self.reward_est = np.zeros(self.num_arms)
        self.pull_cnter = np.zeros(self.num_arms)
        self.alpha = 2
        self.V = np.identity(self.d)
        self.b = np.atleast_2d(np.zeros(self.d)).T # Equivalently => np.zeros(self.d)[:,None]

    def choose_arm(self):
        """Compute UCB scores and return the selected arm and its index"""
        ucb_idx_lst = []
        V_inv = np.linalg.inv(self.V)
        self.theta = np.dot(V_inv, self.b)
        for arm_idx in range(self.num_arms):
            x = self.arms[arm_idx][:, None]
            ucb_idx = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(V_inv, x)))
            ucb_idx_lst.append(ucb_idx)

        selected_arm_idx = np.array(ucb_idx_lst).argmax()
        return self.arms[selected_arm_idx], selected_arm_idx

    def update(self, reward, arm_idx):  # update the parameters
        self.reward_est[arm_idx] += reward
        self.pull_cnter += 1
        self.reward_history.append(reward)
        X = self.arms[arm_idx][:, None]
        self.V += np.dot(X, X.T)
        self.b += reward * X


def regret_vs_horizon(REPEAT, HORIZON):
    LinUCB_history = np.zeros(HORIZON)
    my_context_arm = context_arm()
    for _ in tqdm(range(REPEAT)):
        LinUCB_instance = LinUCB(available_arms)
        for i in range(HORIZON):
            arm, arm_idx = LinUCB_instance.choose_arm()
            reward = my_context_arm.pull_arm(arm_idx)
            LinUCB_instance.update(reward, arm_idx)
        LinUCB_history += np.array(LinUCB_instance.reward_history)

    LinUCB_expected_reward = LinUCB_history / REPEAT
    LinUCB_expected_reward = np.cumsum(LinUCB_expected_reward)
    best_rewards = my_context_arm.genie_reward()
    best_rewards = best_rewards * np.linspace(1, HORIZON, num=HORIZON)
    LinUCB_regret = best_rewards - LinUCB_expected_reward
    return LinUCB_regret

REPEAT = 5 #500
HORIZON = 10 #10000
LinUCB_regret = regret_vs_horizon(REPEAT, HORIZON)

_plot(LinUCB_regret, log_x_axis=True)
zz = -1