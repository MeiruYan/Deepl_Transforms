#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[ ]:

# Extract method name and sample size dynamically
dataset_path = "path"
method_name = os.path.basename(os.path.dirname(dataset_path))
sample_size = os.path.splitext(os.path.basename(dataset_path))[0].split('_')[-1]

# Read the data
data = pd.read_csv(dataset_path)

def split_and_encode(peptide_sequence):
    encoded_sequence = []
    for char in peptide_sequence:
        if char.isalpha():
            encoded_sequence.append(ord(char))
        else:
            encoded_sequence.append(ord(char))
    return encoded_sequence


data['encoded_pep'] = data['pep'].apply(lambda x: split_and_encode(x))
X = pd.DataFrame(data['encoded_pep'].tolist())

target = data[['lable1', 'lable2', 'lable3']]
encoding_dict = {}
for index, row in data.iterrows():
    original_sequence = row['pep']
    encoded_sequence = row['encoded_pep']
    for original_char, encoded_char in zip(original_sequence, encoded_sequence):
        if original_char not in encoding_dict:
            encoding_dict[original_char] = encoded_char

# 准备训练数据
X = np.array(X)
y = np.array(target)

scaler_X = MinMaxScaler()
scaler_X.fit(X)
sc_X = scaler_X.transform(X)

scaler_y = MinMaxScaler()
scaler_y.fit(y)
sc_y = scaler_y.transform(y)
length = sc_y.shape[0]
train_length = int(0.8 * length)
val_length = int(0.9 * length)

x_trainval, x_test, y_trainval, y_test = train_test_split(sc_X, sc_y, test_size=0.2, random_state=2024)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=2024)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

# 设置超参数
input_dim = x_train.shape[1]  # 输入特征数量
output_dim = y_train.shape[1]  # 输出目标数量
num_heads = 4  # 多头注意力头数
num_layers = 2  # Transformer 层数
hidden_dim = 64  # 隐藏层维度
dropout = 0.1  # Dropout 概率
num_epochs = 100  # 训练轮数
batch_size = 32  # 批次大小
learning_rate = 1e-3  # 学习率


# 定义 Transformer 回归模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 增加序列维度 (batch_size, seq_len, features)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 对所有序列位置求均值
        x = self.fc(x)
        return x


# 初始化模型、损失函数和优化器
model = TransformerRegressor(input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 训练模型并保存最佳模型
best_val_loss = float('inf')
best_model_state = None
losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_outputs = model(val_x)
            val_loss += criterion(val_outputs, val_y).item()
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')
    losses.append(float(loss))
    val_losses.append(float(val_loss))
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

# 加载最佳模型
model.load_state_dict(best_model_state)

# 测试模型
model.eval()
with torch.no_grad():
    test_predictions = model(x_test)

plot_path = f"path"
ensure_dir(plot_path)
plt.figure(figsize=(12, 4), dpi=200)
plt.plot(range(num_epochs), losses, label='loss')
plt.plot(range(num_epochs), val_losses, label='val_loss')
plt.grid()
plt.legend()
plt.savefig(plot_path, dpi=300)
plt.show()

import matplotlib.pyplot as plt

inv_test_y = scaler_y.inverse_transform(y_test)
inv_train_y = scaler_y.inverse_transform(y_train)

inv_pred_test_y = scaler_y.inverse_transform(test_predictions.numpy())
inv_pred_train_y = scaler_y.inverse_transform(model(torch.tensor(x_train, dtype=torch.float32)).detach().numpy())

metrics = []
for i in range(y_test.shape[1]):
    print("====" * 10)
    mse = mean_squared_error(inv_test_y[:, i], inv_pred_test_y[:, i])
    mae = mean_absolute_error(inv_test_y[:, i], inv_pred_test_y[:, i])
    r2 = r2_score(inv_test_y[:, i], inv_pred_test_y[:, i])
    ev = explained_variance_score(inv_test_y[:, i], inv_pred_test_y[:, i])

    train_mse = mean_squared_error(inv_train_y[:, i], inv_pred_train_y[:, i])
    train_mae = mean_absolute_error(inv_train_y[:, i], inv_pred_train_y[:, i])
    train_r2 = r2_score(inv_train_y[:, i], inv_pred_train_y[:, i])
    train_ev = explained_variance_score(inv_train_y[:, i], inv_pred_train_y[:, i])

    metrics.append([target.columns[i], mse, mae, r2, ev, train_mse, train_mae, train_r2, train_ev])





# In[27]:


# x_train.shape, x_test.shape, x_val.shape


# In[ ]:




