# -*- coding:utf-8 -*-
import math
import os
import sys
import time

import gym
import numpy as np

##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #


class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table as a dictionary
        self.q_table = {}

        # Store current state, action for SARSA update
        self.state = None
        self.action = None

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # Initialize Q-values for new state
        if observation not in self.q_table:
            self.q_table[observation] = np.zeros(len(self.all_actions))

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: choose random action
            action = np.random.choice(self.all_actions)
        else:
            # Exploitation: choose best action
            action = np.argmax(self.q_table[observation])

        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        """learn from experience using SARSA algorithm"""
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.all_actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.all_actions))

        # SARSA update: Q(s,a) = Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]
        if done:
            # If terminal state, there is no next Q-value
            target_q = reward
        else:
            # Use the next action's Q-value (on-policy)
            target_q = reward + self.gamma * self.q_table[next_state][next_action]

        # Update Q-value
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

        return True

    def decay_epsilon(self):
        """Decay epsilon value after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action

    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("(ﾉ｀⊿´)ﾉ")
        return False

    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####


class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action

    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("(ﾉ｀⊿´)ﾉ")
        return False

    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
