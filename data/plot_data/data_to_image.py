import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def smooth(data: pd.Series):
    smooth_data = pd.Series(data).ewm(alpha=0.1).mean()
    return smooth_data

def cumulative_mean(data):
    data = np.array(data)  # 将输入数据转换为numpy数组
    cumulative_sum = np.cumsum(data)  # 计算累积和
    cumulative_mean = cumulative_sum / np.arange(1, len(data) + 1)  # 计算累积均值
    return cumulative_mean

colors = ['#e5071a', '#00CD00', '#1e90ff', '#FF9900', '#fd79a8', '#8074b2', '#636e72']

# -------------------
# algs = ['PPO', 'PPO_B_0', 'PPO_B_500', 'PPO_B_1k']  # * 给出算法文件夹名
algs = ['RDQN_Normal', 'RDQN~cvae_STA~regular']  # * 给出算法文件夹名
mission = 'highway'
target_index = 'Return'
# -------------------

plt.figure(figsize=(8, 5))

for index, alg in enumerate(algs):
    df = pd.DataFrame()
    file_list = None
    for file_name in os.listdir(f'data/plot_data/{mission}/{alg}'): # f'data/sumo_out/{alg}'
        seed = file_name.split('_')[0]
        df[seed] = pd.read_csv(f'data/plot_data/{mission}/{alg}/' + file_name)[target_index]  # system_total_stopped

    ndf = pd.DataFrame()
    ndf['Mean'] = df.mean(axis=1)
    ndf['Max'] = df.max(axis=1)
    ndf['Min'] = df.min(axis=1)
    print(f'{alg} mean of {target_index}:', round(ndf['Mean'].mean(), 3))
    ax = sns.set_theme(style='ticks', font_scale=1.3)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.grid(ls=':', color='grey', lw=1)
    plt.xlabel('Episode($×10^3$s)')
    plt.ylabel(f'{target_index}')
    # plt.ylim((10, 100))
    # plt.xlim((0, 200))

    y = ndf['Mean']
    min_y = ndf['Min']
    max_y = ndf['Max']
    x = range(len(y))
    plt.fill_between(x, smooth(min_y), smooth(max_y), alpha=0.15, color=colors[index], edgecolor=colors[index], linestyle='--', linewidth=1)
    ax = sns.lineplot(x=x, y=smooth(y), label=alg, c=colors[index], lw=2)
    
# plt.axhline(y=42.667, color='gray', linestyle='--', label='Fixed 20s', lw=2)
# plt.axhline(y=70.167, color='gray', linestyle='-', label='Monte Carlo', lw=2)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.legend()
plt.show()
# plt.savefig('C:/Users/Bowen/Desktop/仓库/My_papers/TESCAL/figures/exp-others-queue.pdf', bbox_inches='tight')