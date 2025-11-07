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
    
    Args:
        episode_rewards: List of episode rewards during training
        epsilon_values: List of epsilon values during training
        algorithm_name: Name of the algorithm (e.g., 'SARSA', 'Q-Learning', 'Dyna-Q')
        save_dir: Directory to save the plot (default: 'assets')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot episode rewards
    ax1.plot(episode_rewards, linewidth=1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title(f'{algorithm_name} Training: Episode Reward over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon values
    ax2.plot(epsilon_values, linewidth=1, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon Value')
    ax2.set_title(f'{algorithm_name} Training: Epsilon Decay over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{algorithm_name.lower().replace(" ", "_").replace("-", "_")}_training_metrics.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
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

