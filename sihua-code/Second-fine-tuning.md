## 第二步微调： 增加分类头预测T细胞表位
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from esm.pretrained import esm3_sm_open_v0  # 确保从正确的库导入
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# 定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = EsmSequenceTokenizer()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sequence = self.data_frame.iloc[idx, 0]
        label = int(self.data_frame.iloc[idx, 1])
        tokenized_seq = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.long)
        return tokenized_seq, label

# 定义分类头
class ClassificationHead(nn.Module):
    def __init__(self, input_dim):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 分类数为2
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 从(N, L, C)转换为(N, C, L)以适配卷积层
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, 2)  # 全局平均池化
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 微调模型的函数
def finetune_model(model, classification_head, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classification_head.parameters()), lr=1e-4)

    for epoch in range(10):
        # Train
        model.train()
        classification_head.train()
        total_train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = model(sequences)['representation'][0]
            logits = classification_head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Test
        model.eval()
        classification_head.eval()
        total_test_loss = 0
        correct = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                features = model(sequences)['representation'][0]
                logits = classification_head(features)
                loss = criterion(logits, labels)
                total_test_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_loader)}, Test Loss: {total_test_loss / len(test_loader)}, Accuracy: {correct / len(test_loader.dataset)}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = esm3_sm_open_v0(pretrained_location="ESM3_sm_open_v0-Influenza-A.pth").to(device)
    classification_head = ClassificationHead(input_dim=1280).to(device)

    train_dataset = ClassificationDataset("myclass_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ClassificationDataset("myclass_test.csv")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    finetune_model(base_model, classification_head, train_loader, test_loader, device)

if __name__ == "__main__":
    main()


```
