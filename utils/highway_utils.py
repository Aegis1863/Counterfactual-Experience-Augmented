import os
import sys
import torch
import torch.nn as nn
import time
import wandb
import numpy as np
import pandas as pd
import collections
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import random
import numpy as np
import torch


def read_ckp(ckp_path: str, agent: object, model_name: str, buffer_size: int = 0):
    """读取已有数据, 如果报错, 可以先删除存档"""
    path = "/".join(ckp_path.split('/')[:-1])
    if not os.path.exists(path):  # 检查路径在不在
        os.makedirs(path)
    if os.path.exists(ckp_path):  # 检查文件在不在
        print('\033[34m[ checkpoint ]\033[0m 读取已有模型权重和训练数据...')
        checkpoint = torch.load(ckp_path)
        s_epoch = checkpoint["epoch"]
        s_episode = checkpoint["episode"]

        # 区分算法
        if 'DQN' in model_name:
            agent.q_net.load_state_dict(checkpoint["best_weight"])
        elif 'PPO' in model_name:
            assert not buffer_size, 'PPO 没有经验池!'
            agent.actor.load_state_dict(checkpoint["actor_best_weight"])
            agent.critic.load_state_dict(checkpoint["critic_best_weight"])
        elif 'SAC' in model_name:
            agent.actor.load_state_dict(checkpoint['actor_best_weight'])
            agent.critic_1.load_state_dict(checkpoint['critic_1_best_weight'])
            agent.critic_2.load_state_dict(checkpoint['critic_2_best_weight'])

        return_list = checkpoint["return_list"]
        time_list = checkpoint["time_list"]
        seed_list = checkpoint['seed_list']

        if buffer_size:
            replay_buffer = checkpoint["replay_buffer"]
            return s_epoch, s_episode, return_list, time_list, seed_list, replay_buffer
        return s_epoch, s_episode, return_list, time_list, seed_list
    else:
        print('\033[34m[ checkpoint ]\033[0m 全新训练...')
        if buffer_size:
            return 0, 0, [], [], [], [], [], [], ReplayBuffer(buffer_size)
        return 0, 0, [], [], [], [], [], []


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:  # 逆向折算
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
    return advantage_list


def save_plot_data(return_list, time_list, seed_list, 
                   ckpt_path, seed, pool_size=None):
    system_type = sys.platform  # 操作系统标识
    # ckpt/SAC/big-intersection_42_win32.pt
    alg_name = ckpt_path.split('/')[1]  # 在本项目路径命名中，第二个是算法名
    if not os.path.exists(f"data/plot_data/{alg_name}/"):  # 路径不存在时创建
        os.makedirs(f"data/plot_data/{alg_name}/")  # data/plot_data/SAC/
    log_path = f"data/plot_data/{alg_name}/{seed}_{system_type}.csv"
    return_save = pd.DataFrame()
    return_save["Algorithm"] = [alg_name] * len(return_list)  # 算法名称
    return_save["Seed"] = seed_list
    return_save["Return"] = return_list
    if pool_size:
        return_save["Pool size"] = pool_size
    return_save["Log time"] = time_list
    return_save.to_csv(log_path, index=False, encoding='utf-8-sig')


def train_PPO_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    return_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
):
    """
    同策略, 没有经验池, 仅限演员评论员框架
    """
    start_time = time.time()
    best_score = -1e10  # 初始分数
    if not return_list:
        return_list = []
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            transition_dict = {
                "states": [],
                "actions": [],
                "next_states": [],
                "rewards": [],
                "dones": [],
                "truncated": [],
            }
            episode_return = 0
            state, done, truncated = env.reset()[0], False, False
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(done)
                transition_dict["truncated"].append(truncated)
                state = next_state
                episode_return += reward
            # 记录
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            agent.update(transition_dict)  # 更新参数

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_best_weight]

            # if dynamic_model:
                # PPO_rollout(agent, dynamic_model, transition_dict, roll_size, roll_step)

            # 存档
            save_PPO_data(writer, return_list, time_list, seed_list, 
                          ckpt_path, epoch, episode, best_weight, seed)
            # 记录时间
            episode_time = (time.time() - episode_begin_time) // 60
            # 打印回合信息
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))

            s_episode = 0
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    total_time = time.time() - start_time
    print(f"\033[32m[ 总耗时 ]\033[0m {total_time / 60:.2f}分钟")
    # 如果检查点保存了回报列表, 可以不返回return_list
    return return_list, total_time // 60


def save_PPO_data(writer, return_list, time_list, seed_list, 
                  ckpt_path, epoch, episode, weight, seed):
    # wandb 存档
    actor_best_weight, critic_best_weight = weight
    if writer > 1:  # wandb 存档
        wandb.log({
            "_return_list": return_list[-1],
            })
    if writer > 0:  
        # 训练权重存档
        torch.save(
            {
                "epoch": epoch,
                "episode": episode,
                "actor_best_weight": actor_best_weight,
                "critic_best_weight": critic_best_weight,
                "return_list": return_list,
            },
            ckpt_path,
        )
        # 绘图数据存档
        save_plot_data(return_list, time_list, seed_list, ckpt_path, seed)


def train_SAC_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    replay_buffer: object,
    minimal_size: int,
    batch_size: int,
    return_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
):
    """
    异策略
    """
    start_time = time.time()
    best_score = -1e10  # 初始分数
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            env_pool = ReplayBuffer(5000)  # 环境模型经验
            episode_begin_time = time.time()
            episode_return = 0
            step = 0
            state, done, truncated = env.reset()[0], False, False
            while not (done | truncated):
                # if step % 50 == 0 and step > 0 and dynamic_model:
                #     train_model(dynamic_model, env_pool)
                #     rollout_model(agent, dynamic_model, roll_step, 
                #                   rollout_batch_size, env_pool, replay_buffer)
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                # if dynamic_model and step < 100:
                #     env_pool.add(state, action, reward, next_state, done, truncated)
                
                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:  # 确保先收集到一定量的数据再采样
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                    transition_dict = {
                        "states": b_s, "actions": b_a, "next_states": b_ns,
                        "rewards": b_r, "dones": b_d, "truncated": b_t,
                    }
                    agent.update(transition_dict)
                step += 1
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_1_best_weight = agent.critic_1.state_dict()
                critic_2_best_weight = agent.critic_2.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight,
                               critic_1_best_weight,
                               critic_2_best_weight]
            if writer > 0:  # 存档
                save_SAC_data(writer, replay_buffer, return_list,
                              time_list, seed_list, ckpt_path, epoch, episode, best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))
        s_episode = 0
    env.close()
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic_1.load_state_dict(critic_1_best_weight)
    agent.critic_2.load_state_dict(critic_2_best_weight)
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def save_SAC_data(writer, replay_buffer, return_list, time_list, 
                  seed_list, ckpt_path, epoch, episode, weight, seed):
    actor_best_weight, critic_1_best_weight, critic_2_best_weight = weight
    if writer > 1:  # wandb 存档
        wandb.log({
            "_return_list": return_list[-1],
            "pool_size": replay_buffer.size(),
            })
    # 训练权重存档
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "actor_best_weight": actor_best_weight,
            "critic_1_best_weight": critic_1_best_weight,
            "critic_2_best_weight": critic_2_best_weight,
            "return_list": return_list,
            "time_list": time_list,
            "seed_list": seed_list,
            "replay_buffer": replay_buffer,
        },
        ckpt_path,
    )

    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list, 
                   ckpt_path, seed, replay_buffer.size())


def train_DQN(
        env: object,
        agent: object,
        writer: bool,
        s_epoch: int,
        total_epoch: int,
        s_episode: int,
        total_episodes: int,
        replay_buffer: object,
        minimal_size: int,
        batch_size: int,
        return_list: list,
        time_list: list,
        seed_list: list,
        seed: int,
        ckpt_path: str):
    start_time = time.time()
    best_score = -100  # 初始化最佳分数
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for epoch in range(s_epoch, total_epoch):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            episode_return = 0
            state, done, truncated = env.reset()[0], False, False
            while not done | truncated:
                action = agent.take_action(state)
                max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理, 主要保留前一状态
                max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                next_state, reward, done, truncated, info = env.step(action)

                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns,
                        'rewards': b_r, 'dones': b_d, 'truncated': b_t,
                    }
                    agent.update(transition_dict)
                if episode_return > best_score:
                    best_weight = agent.q_net.state_dict()
                    best_score = episode_return

            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            # 调整epsilon
            agent.epsilon = max(1 - epoch / (total_epoch / 3), 0.01)
            # 保存检查点
            save_DQN_data(writer, replay_buffer, return_list, time_list, 
                          seed_list, ckpt_path, epoch, episode, agent.epsilon,
                          best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))
            s_episode = 0
    env.close()
    agent.q_net.load_state_dict(best_weight)  # 应用最佳权重
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time


def save_DQN_data(writer, replay_buffer, return_list, time_list, 
                  seed_list, ckpt_path, epoch, episode, epsilon,
                  best_weight, seed):
    # wandb 存档
    if writer:
        wandb.log({
            "_return_list": return_list[-1],
            "pool_size": replay_buffer.size(),
            })
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
    save_plot_data(return_list, time_list, seed_list, ckpt_path, seed)


class ReplayBuffer:
    """异策略的经验缓存"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: dict, action: dict, reward: float, next_state: dict, done: dict, truncated: dict):
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        if batch_size > self.size():
            return self.return_all_samples()
        else:
            transitions = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done, truncated = zip(*transitions)
            return np.array(state), action, reward, np.array(next_state), done, truncated

    def size(self):
        return len(self.buffer)
    
    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done, truncated = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated


class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # Mean
        self.fc22 = nn.Linear(128, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        self.latent_dim = latent_dim

    def encode(self, x, c):
        h1 = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h3 = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def generate(self, batch, action_dim):
        '''显示生成效果'''
        import matplotlib.pyplot as plt
        import seaborn as sns
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conditions = torch.randint(0, action_dim, (batch,)).sort()[0]
        if action_dim > 1:
            one_hot_conditions = torch.eye(action_dim)[conditions].to(device)
        with torch.no_grad():
            sample = torch.randn(batch, self.latent_dim).to(device)
            generated = self.decode(sample, one_hot_conditions).cpu()

        plt.figure(figsize=(10, 8))
        plt.rcParams['font.size'] = 14
        ax = sns.heatmap(generated)
        ax.set_yticks(np.arange(len(conditions)) + 0.5)
        label = [i.item() for i in conditions]
        ax.set_yticklabels(label, rotation=0)
        plt.xlabel('state 分量变化')
        plt.ylabel('action')
        plt.show()


def cvae_train(model, diff_state, action, optimizer, test_and_feedback=False):
    '''
    model: cvae模型
    diff_state: 差分状态， diff_state = state[1:, 5:] - state[:-1, 5:]
    action: 动作，必须是 one-hot 形式
    '''
    def cvae_loss(recon_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(recon_x, x) 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    # 整理数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = prepare_data(diff_state, action, test_and_feedback)
    # 训练
    model.train()
    train_loss = 0
    for state, action in train_loader:
        state, action = state.to(device), action.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(state, action)
        loss = cvae_loss(recon_batch, state, mu, logvar)
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
    # 测试
    if test_and_feedback:
        print(f'Train loss: {train_loss/state.shape[0]:.4f}')
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for state, action in test_loader:
                state, action = state.to(device), action.to(device)
                recon_batch, mu, logvar = model(state, action)
                test_loss += cvae_loss(recon_batch, state, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print(f'Test loss: {test_loss:.4f}')
        return train_loss, test_loss

def prepare_data(diff_state, action, test_and_feedback):
    dataset = TensorDataset(diff_state, action)
    batch_size = 64
    train_size = int(0.8 * len(dataset)) if test_and_feedback else len(dataset)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader