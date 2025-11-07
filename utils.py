# -*- coding:utf-8 -*-
# Utility functions for visualization and analysis
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


def plot_training_metrics(episode_rewards, epsilon_values, algorithm_name, save_dir='assets'):
    """
    Plot training metrics (episode rewards and epsilon values) and save to file.
    Creates two visualizations:
    1. Dual-scale training curve (first 90% vs last 10% episodes)
    2. Epsilon decay curve

    Args:
        episode_rewards: List of episode rewards during training
        epsilon_values: List of epsilon values during training
        algorithm_name: Name of the algorithm (e.g., 'SARSA', 'Q-Learning', 'Dyna-Q')
        save_dir: Directory to save the plot (default: 'assets')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    episodes = np.arange(len(episode_rewards))

    # ========== Subplot 1: Dual-Scale Training Curve ==========
    # Split point: 90% for early phase, last 10% for late phase (fine-grained)
    split_point = int(len(episode_rewards) * 0.9)

    # Left Y-axis: First 90% (larger range)
    color_early = 'tab:blue'
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('First 90% Episodes Reward', color=color_early, fontsize=11)
    ax1.plot(episodes[:split_point], episode_rewards[:split_point],
             color=color_early, alpha=0.6, linewidth=1, label='First 90%')
    ax1.tick_params(axis='y', labelcolor=color_early)
    ax1.grid(True, alpha=0.3)

    # Right Y-axis: Last 10% (smaller range, more precise)
    ax1_twin = ax1.twinx()
    color_late = 'tab:red'
    ax1_twin.set_ylabel('Last 10% Episodes Reward (Fine-grained)', color=color_late, fontsize=11)
    ax1_twin.plot(episodes[split_point:], episode_rewards[split_point:],
                  color=color_late, alpha=0.7, linewidth=1.5, label='Last 10%')
    ax1_twin.tick_params(axis='y', labelcolor=color_late)

    # Add vertical line at split point
    ax1.axvline(x=split_point, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(split_point, ax1.get_ylim()[1] * 0.95, f'  Split at episode {split_point}',
             ha='left', va='top', fontsize=10, color='gray', fontweight='bold')

    ax1.set_title(f'{algorithm_name}: Dual-Scale Training Curve (First 90% vs Last 10%)',
                  fontsize=13, fontweight='bold')

    # ========== Subplot 2: Epsilon Decay ==========
    ax2.plot(epsilon_values, linewidth=1.5, color='tab:orange', alpha=0.8)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Epsilon Value', fontsize=11)
    ax2.set_title(f'{algorithm_name}: Epsilon Decay over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Mark key epsilon milestones
    epsilon_min = min(epsilon_values)
    epsilon_max = max(epsilon_values)
    epsilon_half = (epsilon_max + epsilon_min) / 2

    # Find where epsilon reaches half
    half_idx = int(np.argmin(np.abs(np.array(epsilon_values) - epsilon_half)))
    ax2.axhline(y=epsilon_half, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=half_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(half_idx, epsilon_half, f'  Half at episode {half_idx}',
             fontsize=9, color='gray', va='bottom')

    plt.tight_layout()

    # Save figure
    filename = f'{algorithm_name.lower().replace(" ", "_").replace("-", "_")}_training_metrics.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"Training metrics plot saved as '{filepath}'")
    plt.show()


def visualize_path(agent, env, algorithm_name, background_image_path=None, save_dir='assets'):
    """
    Visualize the path taken by the trained agent on the cliff-walking grid.
    
    Args:
        agent: Trained agent with choose_action method
        env: Gym environment
        algorithm_name: Name of the algorithm (e.g., 'SARSA', 'Q-Learning', 'Dyna-Q')
        background_image_path: Optional path to background image
        save_dir: Directory to save the plot (default: 'assets')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Run one episode with greedy policy (epsilon=0)
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # Use greedy policy
    
    s = env.reset()
    path = [s]
    done = False
    
    while not done and len(path) < 100:
        a = agent.choose_action(s)
        s, r, done, info = env.step(a)
        path.append(s)
    
    agent.epsilon = old_epsilon  # Restore epsilon
    
    # Convert state numbers to (x, y) coordinates
    # CliffWalking is a 4x12 grid (4 rows, 12 columns)
    # State number = row * 12 + col
    coords = [(state % 12, 3 - state // 12) for state in path]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Try to load background image
    if background_image_path and os.path.exists(background_image_path):
        try:
            img = Image.open(background_image_path)
            # Convert PIL image to numpy array for matplotlib
            img_array = np.array(img)
            ax.imshow(img_array, extent=[0, 12, 0, 4], aspect='auto', alpha=0.7)
        except Exception as e:
            print(f"Could not load background image: {background_image_path}, Error: {e}")
    
    # Draw grid
    for i in range(13):
        ax.axvline(i, color='gray', linewidth=0.5)
    for i in range(5):
        ax.axhline(i, color='gray', linewidth=0.5)
    
    # Mark special cells
    # Start (bottom-left)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=True, color='green', alpha=0.3))
    ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Goal (bottom-right)
    ax.add_patch(Rectangle((11, 0), 1, 1, fill=True, color='gold', alpha=0.3))
    ax.text(11.5, 0.5, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Cliff (bottom row, excluding start and goal)
    for i in range(1, 11):
        ax.add_patch(Rectangle((i, 0), 1, 1, fill=True, color='red', alpha=0.3))
        ax.text(i + 0.5, 0.5, 'C', ha='center', va='center', fontsize=12, color='darkred')
    
    # Draw path
    x_coords = [c[0] + 0.5 for c in coords]
    y_coords = [c[1] + 0.5 for c in coords]
    ax.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=6, label='Agent Path', alpha=0.7)
    
    # Mark start and end of path
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, label='End')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'{algorithm_name} Agent Final Path (Length: {len(path)} steps)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{algorithm_name.lower().replace(" ", "_").replace("-", "_")}_final_path.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Final path visualization saved as '{filepath}'")
    plt.show()

