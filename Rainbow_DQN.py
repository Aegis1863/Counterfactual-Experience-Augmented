'''
https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb
'''

import math
import os
import sys
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.segment_tree import MinSegmentTree, SumSegmentTree
from utils.cvae import CVAE, cvae_train
import seaborn as sns
import sumo_rl
import time
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DQN 任务')
parser.add_argument('--model_name', default="highway_RDQN", type=str, help='模型名称, 任务_模型')
parser.add_argument('--symbol', default='Normal', type=str, help='特殊唯一标识')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-w', '--writer', default=1, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-e', '--step', default=15000, type=int, help='运行回合数')
parser.add_argument('-b', '--buffer_size', default=20000, type=int, help='经验池大小')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=45, type=int, help='结束种子')
args = parser.parse_args()

def save_DQN_data(replay_buffer, return_list, time_list, pool_list,
                  seed_list, ckpt_path, epoch, episode, epsilon,
                  best_weight, seed):
    path = "/".join(ckpt_path.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)
    # 训练权重存档
    torch.save({
        'epoch': epoch,
        'episode': episode,
        'best_weight': best_weight,
        'epsilon': epsilon,
        "return_list": return_list,
        "time_list": time_list,
        "seed_list": seed_list,
        "replay_buffer": replay_buffer,
    }, ckpt_path)

    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list, ckpt_path, seed, pool_list)

def save_plot_data(return_list, time_list, seed_list, ckpt_path, seed, pool_size=None):
    system_type = sys.platform  # 操作系统标识
    # ckpt/SAC/big-intersection_42_win32.pt
    mission_name = ckpt_path.split('/')[1]
    alg_name = ckpt_path.split('/')[2]  # 在本项目路径命名中，第二个是算法名
    file_path = f"data/plot_data/{mission_name}/{alg_name}"  # data/plot_data/highway/SAC/
    if not os.path.exists(file_path):  # 路径不存在时创建
        os.makedirs(file_path)
    log_path = f"{file_path}/{seed}_{system_type}.csv"
    return_save = pd.DataFrame()
    return_save["Algorithm"] = [alg_name] * len(return_list)  # 算法名称
    return_save["Seed"] = seed_list
    return_save["Return"] = return_list
    if pool_size:
        return_save["Pool size"] = pool_size
    return_save["Log time"] = time_list
    return_save.to_csv(log_path, index=False, encoding='utf-8-sig')

def counterfactual_exp_expand(replay_buffer, sta, batch_size, action_space_size, distance_ratio):
    '''
    replay_buffer: 经验池
    sta: cvae
    batch_size: 抽多少经验
    action_space_size: 动作空间大小
    distance_threshold: 经验差距阈值，差距太大的匹配经验被放弃
    '''
    # 抽样 batch_size 组真实经验
    samples = replay_buffer.sample_new_real_exp(batch_size)
    b_s, b_ns, b_a, b_r, b_d = samples['obs'], samples['next_obs'], samples['acts'], samples['rews'], samples['done']
    b_s, b_ns, b_a, b_r, b_d = [torch.tensor(i) for i in [b_s, b_ns, b_a, b_r, b_d]]
    
    # 生成反事实动作和其独热向量表示
    counterfactual_actions = []
    for a in b_a:
        counterfactual_actions.append([i for i in range(action_space_size) if i != a])
    counterfactual_actions = torch.tensor(counterfactual_actions).flatten()

    one_hot_cf_actions = torch.nn.functional.one_hot(
        counterfactual_actions, num_classes=action_space_size)

    # 生成反事实状态转移向量
    diff_state = sta.inference(one_hot_cf_actions)

    # 扩展状态以匹配反事实状态转移
    expand_b_s = b_s.repeat_interleave(action_space_size - 1, dim=0)
    b_ns_prime = expand_b_s + diff_state

    # 读取所有真实经验
    all_samples = replay_buffer.retrieve_real_experiences()
    all_ns, all_r = torch.tensor(all_samples['next_obs']), torch.tensor(all_samples['rews'])
    
    # 将真实经验和虚拟经验拼接成向量
    # real_exp = torch.cat((all_s, torch.nn.functional.one_hot(all_a, num_classes=action_space_size), all_ns), dim=1)
    # fake_exp = torch.cat((expand_b_s, one_hot_actions, b_ns_prime), dim=1)
    
    # 计算虚拟经验与真实经验的距离并找到最匹配的真实经验
    # distances = torch.cdist(fake_exp, real_exp)
    distances = torch.cdist(b_ns_prime, all_ns)
    min_indices = torch.argmin(distances, dim=1)
    min_distances = distances[torch.arange(distances.size(0)), min_indices]
    k = int(len(min_distances) * distance_ratio)


    # 筛选出距离小于阈值的虚拟经验
    _, sorted_indices = torch.sort(min_distances)
    close_matches = sorted_indices[:k]
    valid_min_indices = min_indices[close_matches]
    
    valid_fake_s = expand_b_s[close_matches].numpy()
    valid_fake_r = all_r[valid_min_indices].numpy()
    valid_fake_a = one_hot_cf_actions[close_matches].argmax(dim=1).numpy()
    valid_fake_ns = b_ns_prime[close_matches].numpy()
    # 虚拟经验的其他标记
    b_d_prime = np.zeros_like(valid_fake_r)
    # 更新经验池
    for s, a, r, ns, d in zip(valid_fake_s, valid_fake_a, valid_fake_r, valid_fake_ns, b_d_prime):
        replay_buffer.store(s, a, r, ns, d, 1, 0)  # 最后两个数字含义：虚拟经验，未被反事实抽样
    return replay_buffer


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.exp_type_buf = np.zeros(size, dtype=np.float32)  # 虚拟经验是 1，真实经验是 0
        self.cf_sped_buf = np.zeros(size, dtype=np.float32)  # 没被抽的是 0 , cf_sped: counterfactual_sampled
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.capacity = size
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
        exp_type: bool,
        cf_sped: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done, exp_type, cf_sped)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.exp_type_buf[self.ptr] = exp_type
        self.cf_sped_buf[self.ptr] = cf_sped
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            exp_type = self.exp_type_buf[idxs],
            cf_sped = self.cf_sped_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            exp_type=self.exp_type_buf[idxs],
            cf_sped=self.cf_sped_buf[idxs],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-5:-2]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-5:-2]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    
        
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
        exp_type: bool,
        cf_sped: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done, exp_type, cf_sped)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta >= 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        exp_type = self.exp_type_buf[indices]
        cf_sped = self.cf_sped_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            exp_type=exp_type,
            cf_sped=cf_sped,
            weights=weights,
            indices=indices,
        )
    
    def retrieve_real_experiences(self) -> Dict[str, np.ndarray]:
        """采样真实经验，即exp_type值为0的"""
        assert len(self) > 0

        indices = np.where(self.exp_type_buf==0)[0][:self.size-1]
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
        )
    
    def sample_new_real_exp(self, batch_size):
        '''采样未经反事实推断的真实经验，即exp_type和sped_buf值同时为0的'''
        assert len(self) > 0
        indices = np.where((self.exp_type_buf==0) & (self.cf_sped_buf==0))[0][:self.size-1]
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        self.cf_sped_buf[indices] = 1  # 标记为已经采样过
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # * CVAE
        distance_threshold: float = 0.2,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = torch.multiply(*env.observation_space.shape)
        self.action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # * CVAE
        if args.sta:
            self.distance_threshold = distance_threshold  # ! 控制虚拟经验与真实经验的初始差距
            if args.sta_kind:  # 读取预训练模型
                print(f'==> 读取{args.sta_kind} cvae模型')
                path = f'model/cvae/{mission}/{args.sta_kind}.pt'
                self.sta = torch.load(path, map_location=self.device)
            else:
                print(f'==> 在线训练 cvae模型')
                self.sta = CVAE(obs_dim, self.action_dim, obs_dim)  # 在线训练
        else:
            self.sta = None
            self.distance_threshold = None
        self.total_step = 0
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, self.action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(obs_dim, self.action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        state = state.reshape(-1)
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # ! 反事实经验拓展 # TODO
        if self.sta and self.total_step % (2 * self.batch_size) == 0 and self.total_step > 1 and self.memory.size < self.memory.capacity:
            self.memory = counterfactual_exp_expand(self.memory, self.sta, self.batch_size, self.action_dim, self.distance_threshold)
            # self.distance_threshold = max(self.distance_threshold * (self.memory.size - self.memory.capacity)**2 / self.memory.capacity**2, 0.05)
        
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.total_step += 1
        next_state = next_state.reshape(-1)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done, 0, 0]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        losses = []
        scores = []
        time_list = []
        seed_list = []
        pool_list = []  # 经验池大小
        best_score = -1e10  # 初始化最佳分数
        score = 0
        with tqdm(total=num_frames, mininterval=100, ncols=100) as pbar:
            for frame_idx in range(1, num_frames + 1):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                
                # NoisyNet: removed decrease of epsilon
                
                # PER: increase beta
                fraction = min(frame_idx / num_frames, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if episode ends
                if done:
                    state, _ = self.env.reset(seed=self.seed)
                    scores.append(score)
                    score = 0
                    pbar.set_postfix({
                        # 'Step': num_frames // 400 + 1,
                        'scores': round(np.mean(scores[-20:]), 2),
                        'Pool size': len(self.memory)
                    })
                    time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
                    seed_list.append(self.seed)
                    pool_list.append(len(self.memory))
                # if training is ready
                if len(self.memory) >= self.batch_size * 2:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1
                    
                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()
                
                if score > best_score:
                    best_weight = agent.dqn.state_dict()
                    best_score = score
                
                # 其他记录信息
                pbar.update(1)
            # 保存数据
            save_DQN_data(self.memory, scores, time_list, pool_list, seed_list, CKP_PATH, 
                            0, frame_idx, 0, best_weight, seed)
        self._plot(frame_idx, scores, losses)
        self.env.close()
        return scores, losses
        
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.xlabel('Step')
        plt.ylabel('Return')
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        # plt.show()
        file_path = f'image/tmp/{mission}/{args.symbol}'
        os.makedirs(file_path) if not os.path.exists(file_path) else None
        plt.savefig(f'{file_path}/{self.seed}_{model_name}_{system_type}.pdf')

if __name__ == '__main__':
    seed = 42
    # environment

    env = gym.make('highway-fast-v0')
    env.configure({
        "lanes_count": 4,
        "vehicles_density": 2,
        "duration": 100,
    })

    def seed_torch(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # parameters
    memory_size = args.buffer_size
    batch_size = 128
    target_update = 100

    mission = args.model_name.split('_')[0]
    model_name = args.model_name.split('_')[1]

    # VAE
    # --------- 调试用 --------
    if sys.platform != 'linux':
        args.sta = True
        args.sta_kind = 'regular'
        args.symbol = args.sta_kind
    # ------------------------
    args.model_name = args.model_name + '~' + 'cvae' if args.sta else args.model_name
    
    # 其他
    system_type = sys.platform  # 操作系统
    begin_time = time.time()
    
    # 训练
    for seed in range(args.begin_seed, args.end_seed + 1):
        seed_torch(seed)
        CKP_PATH = f'ckpt/{"/".join(args.model_name.split("_"))}_{args.symbol}/{seed}/{system_type}.pt'
        # train
        agent = DQNAgent(env, memory_size, batch_size, target_update, seed, distance_threshold=0.2, n_step=1)
        scores, losses = agent.train(args.step)
        
        train_time = (time.time() - begin_time) / 60
        print('当前花费总时间: %.2f min'%train_time)