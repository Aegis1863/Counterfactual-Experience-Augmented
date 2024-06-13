'''
dueling DQN
'''

import os
import sys
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from utils.highway_utils import read_ckp, train_DQN
from utils.cvae import CVAE, cvae_train
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DQN 任务')
parser.add_argument('--model_name', default="highway_DQN~test", type=str, help='模型名称, 任务_模型')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-w', '--writer', default=1, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
parser.add_argument('-e', '--episodes', default=1500, type=int, help='运行回合数')
parser.add_argument('-b', '--buffer_size', default=30000, type=int, help='经验池大小')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=42, type=int, help='结束种子')
args = parser.parse_args()

if args.writer == 2:
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:  # 该文件中写入一行wandb的API
            api_key = f.read()
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = "offline"

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.h_1(F.relu(self.fc1(x)))))
        V = self.fc_V(F.relu(self.h_1(F.relu(self.fc1(x)))))
        Q = V + A - A.mean(-1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
    
class DQN:
    ''' DQN算法,包括Double DQN '''
    
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, epsilon, update_interval, sta, device,):
        
        self.action_dim = action_dim
        self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval
        self.sta = sta
        self.sta = args.sta_kind
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        
        # * 技巧一
        if self.sta and self.sta.quality > 0.3:
            pre_next_state = self.predict_next_state(states, actions, next_states)
            target_q1 = self.target_q_net(pre_next_state).detach()
            target_q2 = self.target_q_net(next_states).detach()
            max_next_q_values = torch.min(target_q1, target_q2)
        else:
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones | truncated)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step() # 执行Adam梯度下降

        if self.count % self.update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
    
    def train_cvae(self, state, action, next_state, test_and_feedback, batch_size):
        vae_action = action
        diff_state = next_state - state
        loss = cvae_train(self.sta, self.device, diff_state, vae_action, self.sta_optimizer, test_and_feedback, batch_size)
        return loss
    
    def predict_next_state(self, state, action, next_state):
        with torch.no_grad():
            sample = torch.randn(state.shape[0], 32).to(device)  # 随机采样的
            generated = self.sta.decode(sample, action)
        pre_next_state = torch.concat([next_state, state + generated], dim=-1)
        return pre_next_state
    

if __name__ == '__main__':     
    # * --------------------- 参数 -------------------------
    # 环境相关
    env = gym.make('highway-fast-v0')
    env.configure({
        "lanes_count": 3,
        "vehicles_density": 2,
        "duration": 100,
        "collision_reward": -30,
        "right_lane_reward": 0,
        "high_speed_reward": 0.1,
    })
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mission = args.model_name.split('_')[0]
    model_name = args.model_name.split('_')[1]
    
    # DQN相关
    total_epoch = 1  # 迭代数, 无需多次迭代
    gamma = 0.98
    epsilon = 1  # 刚开始随机动作,更新中线性降低
    update_interval = 50  # 若干回合更新一次目标网络
    minimal_size = 500  # 最小经验数
    batch_size = 256

    # 神经网络相关
    lr = 2e-3
    state_dim = torch.multiply(*env.observation_space.shape)
    hidden_dim = 256
    action_dim = env.action_space.n

    # VAE
    if args.sta:
        args.model_name = args.model_name + '~' + 'cvae'
        if args.sta_kind and args.sta:  # 读取预训练模型
            print(f'==> 读取{args.sta_kind} cvae模型')
            args.model_name = args.model_name + '~' + args.sta_kind
            cvae = torch.load(f'model/cvae/{mission}/{args.sta_kind}.pt', map_location=device)
        else:
            print(f'==> 在线训练 cvae模型')
            cvae = CVAE(state_dim, action_dim, state_dim)  # 在线训练
    else:
        cvae = None

    # 任务相关
    system_type = sys.platform  # 操作系统
    print('device:', device)

    # * ----------------------- 训练 ----------------------------

    for seed in range(args.begin_seed, args.end_seed + 1):
        CKP_PATH = f'ckpt/{"/".join(args.model_name.split('_'))}/{seed}/{system_type}.pt'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, update_interval, cvae, device)

        (s_epoch, s_episode, return_list, 
        time_list, seed_list, replay_buffer) = read_ckp(CKP_PATH, agent,  args.model_name,  args.buffer_size)

        if args.writer > 1:
            wandb.init(
                project="MBPO-SUMO",
                group=args.model_name,
                name=f"{seed}",
                config={
                "episodes": args.episodes,
                "seed": seed,
                "mission name": args.model_name
                }
            )
        
        return_list, train_time = train_DQN(env, agent, args.writer, s_epoch, total_epoch, s_episode,
                                            args.episodes, replay_buffer, minimal_size, 
                                            batch_size, return_list, time_list, seed_list, seed, 
                                            CKP_PATH, args.model_name)

        # * ----------------------- 绘图 ----------------------------

        sns.lineplot(return_list)
        plt.title(f'{args.model_name}, training time: {train_time} min')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.savefig(f'image/tmp/{mission}_{args.model_name}_{system_type}.pdf')
        plt.close()