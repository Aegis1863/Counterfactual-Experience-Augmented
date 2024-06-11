import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # Mean
        self.fc22 = nn.Linear(128, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

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
    
def cvae_loss(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def cvae_train(model, diff_state, action, optimizer, test_and_feedback=False):
    '''
    model: cvae模型
    diff_state: 差分状态， diff_state = state[1:, 5:] - state[:-1, 5:]
    action: 动作，必须是 one-hot 形式
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TensorDataset(diff_state, action)
    batch_size = 64
    train_size = int(0.8 * len(dataset)) if test_and_feedback else len(dataset)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    return train_loss


if __name__ == '__main__':
    data = torch.load('data/dataset/Buffer_of_RDQN.pt')
    state = torch.tensor(data.obs_buf)
    next_state = torch.tensor(data.next_obs_buf)
    actions = torch.tensor(data.acts_buf)

    action = state[1:, :4]
    diff_state = state[1:, 5:] - state[:-1, 5:]

    dataset = TensorDataset(diff_state, action)
    batch_size = 64
    train_size = int(0.8 * len(dataset))  # 80%作为训练集
    test_size = len(dataset) - train_size  # 剩余20%作为测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = diff_state.shape[-1]
    condition_dim = action.shape[-1]
    latent_dim = input_dim

    # 训练
    model = CVAE(input_dim, condition_dim, latent_dim).to(device)
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in trange(1, num_epochs + 1):
        cvae_train(model, diff_state, action, optimizer, True)

    # 绘图

    batch = 32
    # 随机选择条件标签并生成相应的独热编码
    conditions = torch.randint(0, 4, (batch,))
    one_hot_conditions = torch.eye(4)[conditions].to(device)

    with torch.no_grad():
        sample = torch.randn(batch, latent_dim).to(device)
        generated = model.decode(sample, one_hot_conditions).cpu()

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 14
    ax = sns.heatmap(generated)
    ax.set_yticks(np.arange(len(conditions)) + 0.5)
    label = [i.item() for i in conditions]
    ax.set_yticklabels(label, rotation=0)
    plt.xlabel('state 分量变化')
    plt.ylabel('action')
    plt.show()