'''
Multi-armed bandit with e-greedy strategy
Tracking a Nonstationary Problem
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import pow
import random


class Bandit:
    def __init__(self, arms, pulls, epsilon):
        self.arms = arms
        self.pulls = pulls
        self.epsilon = epsilon
        self.history = []
        self.best_action_count = []
        self.Q_1 = np.zeros(self.arms)
        self.means = np.zeros(self.arms)
        self.selected_actions = []

        # self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.rewards = [[] for _ in xrange(self.arms)]
        self.t_rewards = [[] for _ in xrange(self.arms)]

    def get_means(self):
        means = np.zeros(self.arms)

        for index, action_rewards in zip(range(len(means)), self.rewards):
            if len(action_rewards) > 0:
                n = len(action_rewards)
                alpha = 1. / n

                means[index] = pow(1 - alpha, n)

                for i, r in zip(range(len(action_rewards)), action_rewards):
                    means += alpha * pow(1 - alpha, n - i) * r

        return means

    def choose_action(self):
        rand = np.random.uniform(0, 1)

        # select action with 1 - epsilon probability
        if rand > self.epsilon:
            # exploit
            # select arm with best estimated reward
            argmax = np.argmax(self.means)
            return argmax
        else:
            # explore
            return randint(0, len(self.rewards) - 1)

    def get_reward(self, action):
        self.true_reward[action] += np.random.randn()
        return self.true_reward[action]

    def save_reward(self, action, reward):
        if len(self.rewards) == 0:
            self.Q_1[action] = reward

        action_rewards = self.rewards[action]

        if len(action_rewards) > 0:
            n = len(action_rewards)

            alpha = 1. / n
            # alpha = 0.1

            mean = pow(1 - alpha, n)

            for i, r in zip(range(len(action_rewards)), action_rewards):
                mean += alpha * pow(1 - alpha, n - i) * r

            self.means[action] = mean

        self.rewards[action].append(reward)

        self.t_rewards[action].append(reward)
        for i in [x for x in xrange(self.arms) if x != action]:
            self.t_rewards[i].append(None)

    def run(self):
        for t in range(self.pulls):
            action = self.choose_action()
            self.selected_actions.append(action)

            self.best_action_count.append(
                np.argmax(self.true_reward) == action)

            reward = self.get_reward(action)
            self.save_reward(action, reward)

            self.history.append(reward)


if __name__ == '__main__':
    # example bandit
    bandit = Bandit(arms=10, pulls=2000, epsilon=0.1)
    bandit.run()

    for arm, reward, true_reward in zip(range(len(bandit.rewards)),
                                        bandit.rewards, bandit.true_reward):
        pulls = len(reward)
        print "Arm {} pulls: {}, true reward: {}". \
            format(arm + 1, pulls, true_reward)

    print "Best arm: {}".format(np.argmax(bandit.true_reward) + 1)

    plt.figure(figsize=(11, 8))
    plt.subplot(211)

    for i, action in zip(range(bandit.arms), bandit.t_rewards):
        plt.plot(action, '.', label="Action {}".format(i + 1))

    plt.ylabel("Rewards")
    plt.xlabel("Steps")
    plt.legend()

    plt.subplot(212)
    plt.plot([a + 1 for a in bandit.selected_actions],
             'r.', label="Selected actions")

    plt.ylabel("Actions")
    plt.xlabel("Steps")
    plt.yticks(range(1, bandit.arms + 1))
    plt.legend()
    plt.savefig('03_rewards.png')
    plt.clf()

    # experiments
    pulls = 5000
    experiments = 200

    epsilons = [0.01, 0.1, 0]

    mean_outcomes = [np.zeros(pulls) for _ in epsilons]
    best_action_count = [np.zeros(pulls) for _ in epsilons]

    for i in range(experiments):
        for index, epsilon in zip(range(len(epsilons)), epsilons):
            bandit = Bandit(arms=10, pulls=pulls, epsilon=epsilon)
            bandit.run()
            mean_outcomes[index] += bandit.history
            best_action_count[index] += bandit.best_action_count

    for index, epsilon in zip(range(len(epsilons)), epsilons):
        mean_outcomes[index] /= experiments
        plt.plot(mean_outcomes[index], label="epsilon: " + str(epsilon))

    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('03_average_reward.png')
    plt.clf()

    for index, epsilon in zip(range(len(epsilons)), epsilons):
        best_action_count[index] /= experiments
        best_action_count[index] *= 100
        plt.plot(best_action_count[index], label="epsilon: " + str(epsilon))
    plt.ylabel("% Optimal action")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('03_optimal_action.png')
