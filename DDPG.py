import random
import gymnasium as gym
import sumo_rl
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import os
os.environ['LIBSUMO_AS_TRACI'] = '1'  # 终端运行加速
import sys
from utils.highway_utils import ReplayBuffer, train_DDPG_agent
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DDPG 任务')
parser.add_argument('--model_name', default="DDPG", type=str, help='基本算法名称')
parser.add_argument('--mission', default="sumo", type=str, help='任务名称')
parser.add_argument('-n', '--net', default="env/big-intersection/big-intersection.net.xml", type=str, help='SUMO路网文件路径')
parser.add_argument('-f', '--flow', default="env/big-intersection/big-intersection.rou.xml", type=str, help='SUMO车流文件路径')
parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-e', '--episodes', default=30, type=int, help='运行回合数')
parser.add_argument('-r', '--reward', default='diff-waiting-time', type=str, help='奖励函数')
parser.add_argument('--begin_time', default=1000, type=int, help='回合开始时间')
parser.add_argument('--duration', default=2000, type=int, help='单回合运行时间')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=45, type=int, help='结束种子')

args = parser.parse_args()

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 这里直接缩放并输出, 而非像PPO中输出均值方差再采样
        return torch.tanh(self.fc2(x))  # 缩放到动作空间


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # * 拼接状态和动作, 似乎是DDPG首次采用, 之前的Q网络只输入状态, 输出动作状态
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
    
class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim,
                 sigma, actor_lr, critic_lr, tau, gamma, device, training=True):
        self.training = training
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差, 均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state: np.ndarray, temperature=1, explore=False, target=False):
        """默认是训练状态, 非目标网络"""
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # 输出每个动作原始值
        logits = self.target_actor(state) if target else self.actor(state)
        if explore:
            # tau 越大越接近原本分布概率抽样，tau 越小越容易选择输出值最高的
            action_probs = F.gumbel_softmax(logits, tau=temperature)
            action_distribution = Categorical(action_probs)
            action = action_distribution.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net):
        '''将target_net往net方向软更新, 每次更新幅度都很小

        参数说明
        ----------
        net : torch.nn.module
        target_net : torch.nn.module
        '''
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)

        actions = torch.nn.functional.one_hot(actions.squeeze(-1), action_dim)
        # 评论员还是时序差分更新, 评论员现在叫Q网络, 但是和之前价值网络一样
        # 不同点是需要输入状态和动作, 动作由演员选择, DQN里面的Q网络不需要输入动作
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones | truncated)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        # 评论员梯度下降
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # * 注意: 演员是梯度策略, 采用梯度上升, 加负号
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 一直更新, 但是缓慢更新
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


# DDPG算法相关
actor_lr = 3e-4
critic_lr = 3e-3
total_episodes = 200
total_epochs = 10
gamma = 0.98
tau = 0.005  # 软更新参数, tau越小更新幅度越小
buffer_size = 20000
minimal_size = 500
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
replay_buffer = ReplayBuffer(buffer_size)

# 环境相关
if args.mission == 'sumo':
    env = gym.make('sumo-rl-v0',
                net_file=args.net,
                route_file=args.flow,
                use_gui=False,
                begin_time=args.begin_time,
                num_seconds=args.duration,
                reward_fn=args.reward,
                sumo_seed=args.begin_seed,
                sumo_warnings=False,
                additional_sumo_cmd='--no-step-log')
else: 
    env = gym.make('highway-fast-v0')
    env.configure({
        "lanes_count": 4,
        "vehicles_density": 2,
        "duration": 100,
    })

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
system_type = sys.platform  # 操作系统

# 神经网络相关
state_dim = env.observation_space.shape[0] if args.mission == 'sumo' else torch.multiply(*env.observation_space.shape)
hidden_dim = 128
action_dim = env.action_space.n

# 其他
# * ----------------------- 训练 ----------------------------
for seed in trange(args.begin_seed, args.end_seed + 1, mininterval=40, ncols=100):
    CKP_PATH = f'{args.mission}/{args.model_name}'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim,
             sigma, actor_lr, critic_lr, tau, gamma, device)
    s_epoch, s_episode, return_list, time_list, seed_list = 0, 0, [], [], []
    return_list, train_time = train_DDPG_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                s_episode, args.episodes, replay_buffer, minimal_size, 
                                                batch_size, return_list, time_list, seed_list,
                                                seed, CKP_PATH,
                                                )