# -*- coding:utf-8 -*-
# Train Dyna-Q in cliff-walking environment
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import logging
import math
import os
import random
import sys
import time
import warnings

import gym
import numpy as np
from tqdm import tqdm

from agent import Dyna_QAgent
from utils import plot_training_metrics, visualize_path

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

##### START CODING HERE #####

# Hyperparameters
LEARNING_RATE = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
PLANNING_STEPS = 10  # Number of planning steps per real experience (Dyna-Q specific)
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 500
LOG_INTERVAL = 100  # Log every N episodes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dyna_q_training.log'),
    ]
)
logger = logging.getLogger(__name__)

# construct the intelligent agent with hyperparameters
agent = Dyna_QAgent(
    all_actions,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    planning_steps=PLANNING_STEPS
)

logger.info(f"Dyna-Q Agent initialized with lr={LEARNING_RATE}, gamma={GAMMA}, epsilon={EPSILON_START}->{EPSILON_MIN}, decay={EPSILON_DECAY}, planning_steps={PLANNING_STEPS}")

# Lists to track training metrics
episode_rewards = []
epsilon_values = []

# start training
logger.info(f"Starting training for {NUM_EPISODES} episodes...")
pbar = tqdm(range(NUM_EPISODES), desc="Dyna-Q", unit="ep")
for episode in pbar:
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()

    # agent interacts with the environment
    for iter in range(MAX_STEPS_PER_EPISODE):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()

        # update the episode reward
        episode_reward += r

        # Dyna-Q: agent learns from experience (s, a, r, s') + planning
        agent.learn(s, a, r, s_, isdone)

        # update state for next iteration
        s = s_

        if isdone:
            break

    # Decay epsilon after each episode
    agent.decay_epsilon()

    # Track metrics
    episode_rewards.append(episode_reward)
    epsilon_values.append(agent.epsilon)

    # Update tqdm progress bar with current metrics
    pbar.set_description(f"Dyna-Q [reward: {episode_reward:>4.0f}]")
    pbar.set_postfix({"epsilon": f"{agent.epsilon:.4f}"})

    # Log to file every LOG_INTERVAL episodes
    if (episode + 1) % LOG_INTERVAL == 0:
        avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} - Avg Reward (last {LOG_INTERVAL}): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

logger.info("Training completed!")

# close the render window after training.
env.close()

# Plot training metrics
plot_training_metrics(episode_rewards, epsilon_values, 'Dyna-Q')

# Visualize the final path
visualize_path(agent, env, 'Dyna-Q', background_image_path='assets/Figure1.jpg')

##### END CODING HERE #####
