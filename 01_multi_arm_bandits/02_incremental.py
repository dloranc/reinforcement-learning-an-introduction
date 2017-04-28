'''
Multi-armed bandit with e-greedy strategy
With incremental implementation of sample averages
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
import random


class Bandit:
    def __init__(self, arms, pulls, epsilon):
        self.arms = arms
        self.action_count = np.zeros(self.arms)
        self.pulls = pulls
        self.epsilon = epsilon
        self.history = []

        self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.rewards = np.zeros(self.arms)

    def choose_action(self):
        rand = np.random.uniform(0, 1)

        if rand > self.epsilon:
            # exploit
            argmax = np.argmax(self.rewards)
            return argmax
        else:
            # explore
            return randint(0, self.arms - 1)

    def get_reward(self, action):
        return self.true_reward[action] + np.random.randn()

    def save_reward(self, action, reward):
        self.action_count[action] += 1
        self.rewards[action] = self.rewards[action] + \
            1. / self.action_count[action] * \
            (reward - self.rewards[action])

    def run(self):
        for t in range(self.pulls):
            action = self.choose_action()
            reward = self.get_reward(action)
            self.save_reward(action, reward)

            self.history.append(reward)


if __name__ == '__main__':
    # example bandit
    bandit = Bandit(arms=10, pulls=2000, epsilon=0.01)
    bandit.run()

    for arm, pulls, true_reward in zip(range(len(bandit.rewards)),
                                        bandit.action_count, bandit.true_reward):
        print "Arm {}\tpulls: {},\ttrue reward: {}". \
            format(arm + 1, int(pulls), true_reward)

    print "Best arm: {}".format(np.argmax(bandit.true_reward) + 1)

    # experiments
    pulls = 1000
    experiments = 2000

    epsilons = [0.01, 0.1, 0]

    mean_outcomes = [np.zeros(pulls) for _ in epsilons]

    for i in range(experiments):
        for index, epsilon in zip(range(len(epsilons)), epsilons):
            bandit = Bandit(arms=10, pulls=pulls, epsilon=epsilon)
            bandit.run()
            mean_outcomes[index] += bandit.history

    for index, epsilon in zip(range(len(epsilons)), epsilons):
        mean_outcomes[index] /= experiments
        plt.plot(mean_outcomes[index], label="epsilon: " + str(epsilon))

    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('02_plot.png')
