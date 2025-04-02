import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from reshape_data import reshape_data
import os
from datetime import datetime

# 自定义数据集类
class TrafficDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 将字符串转换为浮点数（如果需要）
        sample = [float(value) if isinstance(value, str) else value for value in sample]
        
        # 取前4行作为输入，后2行作为输出
        x = torch.tensor([sample[i] for i in range(4)], dtype=torch.float32)
        y = torch.tensor([sample[i] for i in range(4, 6)], dtype=torch.float32)
        
        return x, y

# 使用Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=78, num_heads=3, num_layers=3, dropout=0.01):
        """
        Transformer模型
        
        参数:
        input_dim: 每个时间步的特征维度（6个特征）
        hidden_dim: Transformer模型的隐藏层维度
        num_heads: Transformer的注意力头数
        num_layers: Transformer层数
        dropout: Dropout率
        
        输入形状: [batch_size, 4, 6] - 4个时间步，每个时间步有6个特征
        输出形状: [batch_size, 2, 6] - 预测未来2个时间步
        """
        super(TransformerModel, self).__init__()
        
        # 使用Transformer模块
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Linear(input_dim, 2 * input_dim)
        
    def forward(self, x):
        # 转换输入形状
        x = x.permute(1, 0, 2)
        
        # Transformer处理
        transformer_out = self.transformer(x, x)
        
        # 获取最后一个时间步的输出
        transformer_out_last = transformer_out[-1, :, :]
        
        # 通过输出层
        output = self.fc_out(transformer_out_last)
        
        # 重塑输出
        output = output.view(x.size(1), 2, 6)
        
        return output

# 评估模型性能
def evaluate_model(predictions, actuals):
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    epsilon = 1e-10
    mape = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + epsilon))) * 100
    y_mean = np.mean(actuals)
    ss_total = np.sum((actuals - y_mean) ** 2)
    ss_residual = np.sum((actuals - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total if ss_total != 0 else 0)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

# 训练模型的函数
def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    epoch_info = ""
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        epoch_info += f"{epoch + 1} {train_loss:.6f} {val_loss:.6f}\n"
    
    return train_losses, val_losses, epoch_info

# 主函数
def main():
    df_1580 = pd.read_excel('d:/交通流项目/main/query-hive-1580_processed.xlsx')
    
    df_1580_selected = df_1580.iloc[:, [1, 7, 2, 8, 3, 9]].values.tolist()
    
    data_1580_reshape = reshape_data(df_1580_selected, 6)
    
    data = data_1580_reshape
    
    np.random.shuffle(data)
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    train_dataset = TrafficDataset(train_data)
    val_dataset = TrafficDataset(val_data)
    test_dataset = TrafficDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    model = TransformerModel().to(device)
    
    train_losses, val_losses, epoch_info = train_model(model, train_loader, val_loader, device)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss') 
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f"测试集损失: {test_loss:.6f}")
    
    eval_results = evaluate_model(np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0))
    print("模型评估结果:", eval_results)

if __name__ == "__main__":
    main()