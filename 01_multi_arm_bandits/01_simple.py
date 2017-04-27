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

        self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.rewards = [[] for x in xrange(len(self.true_reward))]

    def get_means(self):
        means = np.zeros(self.arms)

        for index, action_rewards in zip(range(len(means)), self.rewards):
            if len(action_rewards) > 0:
                means[index] = sum(action_rewards) / len(action_rewards)

        return means

    def choose_action(self):
        '''
        e-greedy policy
        '''

        rand = np.random.uniform(0, 1)
        means = self.get_means()

        if rand > self.epsilon:
            # exploit
            argmax = np.argmax(means)
            return argmax
        else:
            # explore
            return randint(0, len(means) - 1)

    def get_reward(self, action):
        return self.true_reward[action] + np.random.randn()

    def save_reward(self, action, reward):
        self.rewards[action].append(reward)

    def run(self):
        for t in range(self.pulls):
            action = self.choose_action()
            reward = self.get_reward(action)
            self.save_reward(action, reward)

            self.history.append(reward)


if __name__ == '__main__':
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
    plt.savefig('plot.png')
