import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from reshape_data import reshape_data

class TrafficDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 将字符串类型转换为数值类型（如果需要）
        sample = [float(value) if isinstance(value, str) else value for value in sample]
        
        # 取前4行作为输入，后2行作为输出
        x = torch.tensor([sample[i] for i in range(4)], dtype=torch.float32)
        y = torch.tensor([sample[i] for i in range(4, 6)], dtype=torch.float32)
        
        return x, y

class ANNModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=12):
        """
        使用ANN实现时间序列预测
        
        参数:
        input_dim: 每个时间步的特征维度 (6个值/列)
        hidden_dim: 隐藏层的维度
        output_dim: 输出层的维度（2个时间步，每个时间步6个特征）
        """
        super(ANNModel, self).__init__()
        
        # 定义ANN结构
        self.fc1 = nn.Linear(input_dim * 4, hidden_dim)  # 4个时间步，每个时间步6个特征
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出2个时间步，每个时间步6个特征
        
    def forward(self, x):
        # x形状: [batch, 4, 6]
        
        batch_size = x.size(0)
        
        # 展平输入: [batch, 4*6] -> [batch, input_dim*4]
        x_flat = x.view(batch_size, -1)
        
        # 通过全连接层
        x = self.fc1(x_flat)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        # 输出层
        output = self.fc3(x)  # [batch, 12]
        
        # 重塑为 [batch, 2, 6]
        output = output.view(batch_size, 2, 6)
        
        return output

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

def train_model(model, train_loader, val_loader, device, epochs=250, lr=0.001):
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

def main():
    # 读取 Excel 文件
    df_1580 = pd.read_excel('d:/交通流项目/main/query-hive-1580_processed.xlsx')
    
    # 提取并重组列
    df_1580_selected = df_1580.iloc[:, [1, 7, 2, 8, 3, 9]].values.tolist()
    
    # 重塑数据
    data_1580_reshape = reshape_data(df_1580_selected, 6)
    
    # 合并数据
    data = data_1580_reshape
    
    # 数据预处理
    np.random.shuffle(data)
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # 创建数据集和数据加载器
    train_dataset = TrafficDataset(train_data)
    val_dataset = TrafficDataset(val_data)
    test_dataset = TrafficDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 检测并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = ANNModel().to(device)
    
    # 训练模型
    train_losses, val_losses, epoch_info = train_model(model, train_loader, val_loader, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()  # 显示图像
    
    # 测试模型
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
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}')
    
    # 使用多种指标评估模型
    metrics = evaluate_model(np.array(predictions), np.array(actuals))
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        epoch_info += f"{metric_name}: {value:.4f}\n"

    # 保存模型
    base_dir = './model_output/'
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    import os
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    model_path = os.path.join(base_dir, f'model_{timestamp}.pth')
    info_path = os.path.join(base_dir, f'training_info_{timestamp}.txt')
    plt_path = os.path.join(base_dir, f'loss_curve_{timestamp}.png')
    with open(info_path, 'w') as f:
        f.write(epoch_info)
    torch.save(model.state_dict(), model_path)
    plt.savefig(plt_path)
    print("模型训练和评估完成!")


if __name__ == '__main__':
    main()