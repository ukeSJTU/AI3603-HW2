# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000, help="the replay memory buffer size"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=1000,
        help="the timesteps it takes to update the target network",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the batch size of sample from the reply memory",
    )
    parser.add_argument(
        "--start-e",
        type=float,
        default=1.0,
        help="the starting epsilon for exploration",
    )
    parser.add_argument(
        "--end-e", type=float, default=0.5, help="the ending epsilon for exploration"
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=10000, help="timestep to start learning"
    )
    parser.add_argument(
        "--train-frequency", type=int, default=4, help="the frequency of training"
    )
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args


def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


class QNetwork(nn.Module):
    """
    comments: Q网络类，用于近似Q值函数 Q(s,a)
    这是一个全连接神经网络，输入是状态s（8维向量），输出是每个动作的Q值（4个动作）
    网络结构：输入层(8) -> 隐藏层1(120) -> ReLU -> 隐藏层2(84) -> ReLU -> 输出层(4)

    Args:
        env: gym环境对象，用于获取状态空间和动作空间的维度
    """

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),  # 输入层：状态维度 -> 120
            nn.ReLU(),  # 激活函数
            nn.Linear(120, 84),  # 隐藏层：120 -> 84
            nn.ReLU(),  # 激活函数
            nn.Linear(84, env.action_space.n),  # 输出层：84 -> 动作数量（4个动作）
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    comments: epsilon线性衰减调度函数
    用于实现epsilon-greedy策略中epsilon值的线性衰减
    从start_e开始，在duration步内线性衰减到end_e

    Args:
        start_e: 起始epsilon值（如0.3）
        end_e: 最终epsilon值（如0.05）
        duration: 衰减持续的时间步数
        t: 当前时间步

    Returns:
        当前时间步对应的epsilon值
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """
    comments: 设置随机种子以确保实验可复现性
    为Python random、NumPy、PyTorch设置相同的随机种子
    设置cudnn为确定性模式，确保GPU运算结果可复现
    选择计算设备（优先使用GPU，否则使用CPU）
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    comments: 创建LunarLander-v2环境
    使用make_env函数构建环境，该函数会包装环境以记录episode统计信息
    """
    envs = make_env(args.env_id, args.seed)

    """
    comments: 初始化Q网络和目标网络（Target Network）
    - q_network: 主Q网络，用于选择动作和训练更新
    - target_network: 目标Q网络，用于计算TD目标值，提供稳定的学习目标
    - optimizer: Adam优化器，用于更新q_network的参数
    两个网络初始参数相同，但target_network更新频率较低（每500步更新一次）
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """
    comments: 初始化经验回放缓冲区（Replay Buffer）
    用于存储智能体与环境交互的经验 (s, a, r, s', done)
    经验回放可以打破数据的时间相关性，提高样本利用效率
    buffer_size=10000表示最多存储10000条经验
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """
    comments: 开始训练主循环
    重置环境获得初始观测状态
    在total_timesteps（500000步）内进行训练
    """
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        """
        comments: 计算当前时间步的epsilon值
        使用线性衰减策略：在前10%的训练步数内，epsilon从0.3线性衰减到0.05
        这样可以在训练初期多探索，后期多利用已学到的策略
        """
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        """
        comments: epsilon-greedy策略选择动作
        以epsilon概率随机选择动作（探索）
        以1-epsilon概率选择Q值最大的动作（利用）
        这是DQN中平衡探索与利用的关键策略
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

        """
        comments: 执行选定的动作，与环境交互
        获得下一个状态next_obs、奖励rewards、是否结束dones、额外信息infos
        如果episode结束（dones=True），记录该episode的总回报和长度到tensorboard
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training

        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar(
                "charts/episodic_return", infos["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", infos["episode"]["l"], global_step
            )

        """
        comments: 将经验(s, a, r, s', done)存储到经验回放缓冲区
        这些经验将在后续训练中被随机采样使用
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)

        """
        comments: 更新当前状态
        如果episode未结束，使用next_obs作为新状态
        如果episode结束，重置环境获得新的初始状态
        """
        obs = next_obs if not dones else envs.reset()

        if (
            global_step > args.learning_starts
            and global_step % args.train_frequency == 0
        ):
            """
            comments: 开始训练Q网络（满足两个条件）
            1. global_step > learning_starts (10000)：收集足够的经验后才开始训练
            2. global_step % train_frequency == 0：每10步训练一次，提高效率
            """
            """
            comments: 从经验回放缓冲区随机采样一个batch的经验
            batch_size=128，随机采样可以打破数据的时间相关性
            """
            data = rb.sample(args.batch_size)

            """
            comments: 计算TD目标值和损失函数
            1. 使用target_network计算下一状态的最大Q值：max_a' Q_target(s', a')
            2. 计算TD目标：y = r + γ * max_a' Q_target(s', a') * (1 - done)
               如果done=1（终止状态），则TD目标就是r
            3. 使用q_network计算当前Q值：Q(s, a)
            4. 计算MSE损失：loss = (y - Q(s,a))^2
            使用torch.no_grad()是因为target_network不需要梯度更新
            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (
                    1 - data.dones.flatten()
                )
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """
            comments: 记录训练指标到tensorboard
            每100步记录一次TD损失和Q值的平均值，用于监控训练过程
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            """
            comments: 反向传播更新Q网络参数
            1. optimizer.zero_grad()：清空之前的梯度
            2. loss.backward()：计算损失函数对参数的梯度
            3. optimizer.step()：使用Adam优化器更新q_network的参数
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            comments: 定期更新目标网络
            每500步将q_network的参数复制到target_network
            这是DQN的关键技巧：使用固定的目标网络提供稳定的学习目标
            如果target_network更新太频繁，会导致训练不稳定
            """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    """close the env and tensorboard logger"""
    envs.close()
    writer.close()
