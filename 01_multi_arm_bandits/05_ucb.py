'''
Multi-armed bandit with e-greedy strategy
With incremental implementation of sample averages
and with Upper-Bound Confidence
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint
import random


class Bandit:
    def __init__(self, arms, pulls, epsilon, c=0):
        self.arms = arms
        self.action_count = np.zeros(self.arms)
        self.pulls = pulls
        self.epsilon = epsilon
        self.history = []
        self.best_action_count = []
        self.selected_actions = []
        self.t = 0
        self.c = c

        self.true_reward = [np.random.randn() for _ in range(self.arms)]
        self.rewards = np.zeros(self.arms)
        self.t_rewards = [[] for _ in xrange(self.arms)]

    def choose_action(self):
        rand = np.random.uniform(0, 1)

        if rand > self.epsilon:
            # exploit
            ucb = self.rewards + \
                self.c * np.sqrt(np.log(self.t + 1) / (self.action_count + 1))

            return np.argmax(ucb)
        else:
            # explore
            return randint(0, self.arms - 1)

    def get_reward(self, action):
        return self.true_reward[action] + np.random.randn()

    def save_reward(self, action, reward):
        self.action_count[action] += 1

        # alpha = 1. / self.action_count[action]
        alpha = 0.1

        self.rewards[action] = self.rewards[action] + \
            alpha * \
            (reward - self.rewards[action])

        self.t_rewards[action].append(reward)
        for i in [x for x in xrange(self.arms) if x != action]:
            self.t_rewards[i].append(None)

    def run(self):
        for t in range(self.pulls):
            self.t += 1
            action = self.choose_action()
            self.selected_actions.append(action)

            self.best_action_count.append(
                np.argmax(self.true_reward) == action)

            reward = self.get_reward(action)
            self.save_reward(action, reward)

            self.history.append(reward)


if __name__ == '__main__':
    # example bandit
    bandit = Bandit(arms=10, pulls=2000, epsilon=0.01, c=2)
    bandit.run()

    for arm, pulls, true_reward in zip(
            range(len(bandit.rewards)),
            bandit.action_count,
            bandit.true_reward):
        print "Arm {}\tpulls: {},\ttrue reward: {}". \
            format(arm + 1, int(pulls), true_reward)

    best_arm = np.argmax(bandit.true_reward) + 1
    print "Best arm: {}".format(best_arm)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.subplot(211)
    ax.set_title('Best arm: {}'.format(best_arm),
                 fontsize=14, fontweight='bold')

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
    plt.savefig('plots/05_rewards.png')

    # experiments
    pulls = 1000
    experiments = 2000

    settings = [[0.1, 0], [0, 2]]

    mean_outcomes = [np.zeros(pulls) for _ in settings]
    best_action_count = [np.zeros(pulls) for _ in settings]

    for i in range(experiments):
        for index, setting in zip(range(len(settings)), settings):
            bandit = Bandit(arms=10, pulls=pulls, epsilon=setting[0],
                            c=setting[1])
            bandit.run()
            mean_outcomes[index] += bandit.history
            best_action_count[index] += bandit.best_action_count

    plt.figure(figsize=(11, 4))
    for index, setting in zip(range(len(settings)), settings):
        mean_outcomes[index] /= experiments
        plt.plot(mean_outcomes[index], label="epsilon: " +
                 str(setting[0]) + ", c=" + str(setting[1]))

    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('plots/05_average_reward.png')

    plt.figure(figsize=(11, 4))
    for index, setting in zip(range(len(settings)), settings):
        best_action_count[index] /= experiments
        best_action_count[index] *= 100
        plt.plot(best_action_count[index], label="epsilon: " +
                 str(setting[0]) + ", c=" + str(setting[1]))
    plt.ylabel("% Optimal action")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig('plots/05_optimal_action.png')
