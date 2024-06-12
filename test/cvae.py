import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    def generate(self, batch, action_space):
        '''显示生成效果'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conditions = torch.tensor([[i] * 8 for i in range(action_space)]).view(-1)
        if action_space > 1:
            one_hot_conditions = torch.eye(action_space)[conditions].to(device)
        with torch.no_grad():
            sample = torch.randn(batch, self.latent_dim).to(device)
            generated = self.decode(sample, one_hot_conditions).cpu()

        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(generated, cbar=False)
        ax.set_yticks(np.arange(len(conditions)) + 0.5)
        label = [i.item() for i in conditions]
        ax.set_yticklabels(label, rotation=0)
        plt.xlabel('State components')
        plt.ylabel('Action')
        plt.savefig(f'image/VAE/{batch_size}/{time.time()}.png')


def cvae_train(model, device, diff_state, action, optimizer, test_and_feedback=False, batch_size=32):
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
    train_loader, test_loader = prepare_data(diff_state, action, test_and_feedback, batch_size)
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

def prepare_data(diff_state, action, test_and_feedback, batch_size):
    dataset = TensorDataset(diff_state, action)
    train_size = int(0.8 * len(dataset)) if test_and_feedback else len(dataset)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader


if __name__ == '__main__':
    state = torch.load('data/dataset/Buffer_of_RDQN.pt')  # 专家数据
    # state = torch.load('data/dataset/Buffer_of_regular.pt')  # 业余数据

    action = state[1:, :4]
    diff_state = state[1:, 5:] - state[:-1, 5:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = diff_state.shape[-1]
    condition_dim = action.shape[-1]
    latent_dim = input_dim
    batch_size = 16

    # 训练
    model = CVAE(input_dim, condition_dim, latent_dim).to(device)
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in trange(num_epochs, leave=False):
        cvae_train(model, device, diff_state, action, optimizer, True, batch_size)
        model.generate(32, 4)