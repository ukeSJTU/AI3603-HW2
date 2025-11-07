# -*- coding:utf-8 -*-
import math
import os
import random
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

    def learn(self, state, action, reward, next_state, done):
        """learn from experience using Q-Learning algorithm (off-policy)"""
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.all_actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.all_actions))

        # Q-Learning update: Q(s,a) = Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]
        if done:
            # If terminal state, there is no next Q-value
            target_q = reward
        else:
            # Use the maximum Q-value of next state (off-policy)
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

        return True

    def decay_epsilon(self):
        """Decay epsilon value after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    ##### END CODING HERE #####


class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, planning_steps=5):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps  # Number of planning steps per real experience

        # Initialize Q-table as a dictionary
        self.q_table = {}

        # Model: stores observed transitions (s, a) -> (r, s')
        self.model = {}

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

    def learn(self, state, action, reward, next_state, done):
        """learn from experience using Dyna-Q algorithm (Q-Learning + Planning)"""
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.all_actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.all_actions))

        # (a) Direct RL: Q-Learning update from real experience
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

        # (b) Model Learning: Store the transition in the model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state, done)

        # (c) Planning: Learn from simulated experiences using the model
        for _ in range(self.planning_steps):
            # Randomly sample a previously observed state
            if len(self.model) == 0:
                break

            # Sample a random state from the model
            sampled_state = random.choice(list(self.model.keys()))

            # Sample a random action that has been taken from this state
            sampled_action = random.choice(list(self.model[sampled_state].keys()))

            # Get the stored transition
            sampled_reward, sampled_next_state, sampled_done = self.model[sampled_state][sampled_action]

            # Initialize Q-values if needed
            if sampled_state not in self.q_table:
                self.q_table[sampled_state] = np.zeros(len(self.all_actions))
            if sampled_next_state not in self.q_table:
                self.q_table[sampled_next_state] = np.zeros(len(self.all_actions))

            # Q-Learning update from simulated experience
            current_q_sim = self.q_table[sampled_state][sampled_action]
            if sampled_done:
                target_q_sim = sampled_reward
            else:
                target_q_sim = sampled_reward + self.gamma * np.max(self.q_table[sampled_next_state])

            self.q_table[sampled_state][sampled_action] = current_q_sim + self.lr * (target_q_sim - current_q_sim)

        return True

    def decay_epsilon(self):
        """Decay epsilon value after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    ##### END CODING HERE #####
