## 第二步微调： 增加分类头预测T细胞表位
```
import torch  # 导入PyTorch库，用于深度学习操作
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化算法模块
from torch.utils.data import DataLoader, Dataset  # 导入数据加载相关的模块
import pandas as pd  # 导入pandas库，用于数据处理
from esm.pretrained import esm3_sm_open_v0  # 从esm库导入预训练模型
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer  # 导入序列分词器

# 定义用于加载和处理分类数据集的类
class ClassificationDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)  # 读取CSV文件
        self.tokenizer = EsmSequenceTokenizer()  # 初始化序列分词器

    def __len__(self):
        return len(self.data_frame)  # 返回数据集的长度

    def __getitem__(self, idx):
        sequence = self.data_frame.iloc[idx, 0]  # 获取序列数据
        label = int(self.data_frame.iloc[idx, 1])  # 获取标签
        tokenized_seq = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.long)  # 对序列进行编码
        return tokenized_seq, label  # 返回编码后的序列和标签

# 定义一个分类头的神经网络类
class ClassificationHead(nn.Module):
    def __init__(self, input_dim):
        super(ClassificationHead, self).__init__()  # 初始化父类
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)  # 定义第一层卷积
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # 定义第二层卷积
        self.fc1 = nn.Linear(128, 64)  # 第一层全连接
        self.fc2 = nn.Linear(64, 32)  # 第二层全连接
        self.fc3 = nn.Linear(32, 2)  # 第三层全连接，输出两个分类
        self.relu = nn.ReLU()  # 定义激活函数

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应卷积层
        x = self.relu(self.conv1(x))  # 第一层卷积和激活
        x = self.relu(self.conv2(x))  # 第二层卷积和激活
        x = torch.mean(x, 2)  # 全局平均池化
        x = self.relu(self.fc1(x))  # 通过全连接和激活
        x = self.relu(self.fc2(x))  # 通过全连接和激活
        x = self.fc3(x)  # 最终输出层
        return x

# 定义微调模型的函数
def finetune_model(model, classification_head, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = optim.Adam(list(model.parameters()) + list(classification_head.parameters()), lr=1e-4)  # 定义优化器

    for epoch in range(10):  # 循环10个训练周期
        # 训练阶段
        model.train()  # 设置模型为训练模式
        classification_head.train()  # 设置分类头为训练模式
        total_train_loss = 0
        for sequences, labels in train_loader:  # 遍历训练数据
            sequences, labels = sequences.to(device), labels.to(device)  # 将数据移动到设备
            optimizer.zero_grad()  # 清空梯度
            with torch.no_grad():  # 不计算梯度
                features = model(sequences)['representation'][0]  # 获取特征表示
            logits = classification_head(features)  # 获取分类结果
            loss = criterion(logits, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            total_train_loss += loss.item()
        
        # 测试阶段
        model.eval()  # 设置模型为评估模式
        classification_head.eval()  # 设置分类头为评估模式
        total_test_loss = 0
        correct = 0
        with torch.no_grad():  # 不计算梯度
            for sequences, labels in test_loader:  # 遍历测试数据
                sequences, labels = sequences.to(device), labels.to(device)  # 将数据移动到设备
                features = model(sequences)['representation'][0]  # 获取特征表示
                logits = classification_head(features)  # 获取分类结果
                loss = criterion(logits, labels)  # 计算损失
                total_test_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()  # 计算准确率
        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_loader)}, Test Loss: {total_test_loss / len(test_loader)}, Accuracy: {correct / len(test_loader.dataset)}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定使用的设备
    base_model = esm3_sm_open_v0(pretrained_location="ESM3_sm_open_v0-Influenza-A.pth").to(device)  # 加载预训练模型
    classification_head = ClassificationHead(input_dim=1280).to(device)  # 初始化分类头

    train_dataset = ClassificationDataset("myclass_train.csv")  # 加载训练数据集
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 创建训练数据加载器
    test_dataset = ClassificationDataset("myclass_test.csv")  # 加载测试数据集
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 创建测试数据加载器

    finetune_model(base_model, classification_head, train_loader, test_loader, device)  # 执行微调

if __name__ == "__main__":
    main()  # 运行主函数

```
