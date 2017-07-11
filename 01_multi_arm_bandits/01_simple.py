'''
Multi-armed bandit with e-greedy strategy
With saving all rewards for each arm
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
import random


class Bandit:
    def __init__(self, arms, pulls, epsilon):
        self.arms = arms
        self.pulls = pulls
        self.epsilon = epsilon
        self.history = []
        self.best_action_count = []

        self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.rewards = [[] for _ in xrange(self.arms)]

    def get_means(self):
        means = np.zeros(self.arms)

        for index, action_rewards in zip(range(len(means)), self.rewards):
            if len(action_rewards) > 0:
                means[index] = sum(action_rewards) / len(action_rewards)

        return means

    def choose_action(self):
        rand = np.random.uniform(0, 1)

        # select action with 1 - epsilon probability
        if rand > self.epsilon:
            # exploit
            means = self.get_means()  # compute all means
            argmax = np.argmax(means) # select arm with best estimated reward
            return argmax
        else:
            # explore
            return randint(0, len(self.rewards) - 1)

    def get_reward(self, action):
        return self.true_reward[action] + np.random.randn()

    def save_reward(self, action, reward):
        self.rewards[action].append(reward)

    def run(self):
        for t in range(self.pulls):
            action = self.choose_action()

            self.best_action_count.append(np.argmax(self.true_reward) == action)

            reward = self.get_reward(action)
            self.save_reward(action, reward)

            self.history.append(reward)


if __name__ == '__main__':
    # example bandit
    bandit = Bandit(arms=10, pulls=2000, epsilon=0.01)
    bandit.run()

    for arm, reward, true_reward in zip(range(len(bandit.rewards)),
                                        bandit.rewards, bandit.true_reward):
        pulls = len(reward)
        print "Arm {} pulls: {}, true reward: {}". \
            format(arm + 1, pulls, true_reward)

    print "Best arm: {}".format(np.argmax(bandit.true_reward) + 1)

    # experiments
    pulls = 1000
    experiments = 2000

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
    plt.savefig('plots/01_average_reward.png')
    plt.clf()

    for index, epsilon in zip(range(len(epsilons)), epsilons):
        best_action_count[index] /= experiments
        best_action_count[index] *= 100
        plt.plot(best_action_count[index], label="epsilon: " + str(epsilon))
    plt.ylabel("% Optimal action")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('plots/01_optimal_action.png')