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

# 使用ReLU激活函数的模型
class CNNReLUModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=6, num_heads=3, num_layers=3, dropout=0):
        super(CNNReLUModel, self).__init__()
        
        # CNN层用于提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # CNN输出的通道数是32，时间步长是4，展平后维度是16 * 4 = 64（特别注意）
        # 将CNN的输出转换为注意力层的输入
        self.fc_cnn = nn.Linear(64, hidden_dim)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 2 * input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN处理
        x_cnn = x.permute(0, 2, 1)  # 调整维度顺序以适应Conv1d (batch, channels, seq_len)
        x_cnn = self.cnn(x_cnn)     # [batch, 32, 6]
        
        # 展平CNN输出
        x_cnn_flat = x_cnn.reshape(batch_size, -1)  # 确保展平后的维度与fc_cnn的输入维度一致
        
        # 转换为注意力层输入
        x_attention = self.fc_cnn(x_cnn_flat).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 注意力处理
        x_attention = x_attention.permute(1, 0, 2)  # 转换为 [1, batch, hidden_dim]
        x_attention, _ = self.attention(x_attention, x_attention, x_attention)
        x_attention = x_attention.permute(1, 0, 2)  # 转换回 [batch, 1, hidden_dim]
        
        # Transformer处理
        x_transformer = self.transformer_encoder(x_attention)
        
        # 输出层
        output = self.fc_out(x_transformer.squeeze(1))  # [batch, 12]
        
        # 重塑为[batch, 2, 6]形状
        output = output.view(batch_size, 2, 6)
        
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
def train_model(model, train_loader, val_loader, device, epochs=120, lr=0.01):
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
    df_1580 = pd.read_excel('d:/traffic project/main/query-hive-1580_processed.xlsx')
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    model = CNNReLUModel().to(device)
    
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
    
    # 定义保存路径
    base_dir = './model_output/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 定义时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 保存最终的预测图
    prediction_plot_path = os.path.join(base_dir, f'prediction_vis_{timestamp}.png')
    plt.savefig(prediction_plot_path)

    print("时间序列预测可视化完成!")
if __name__ == "__main__":
    main()