## 第一步微调：领域适应
```
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from Bio import SeqIO
import torch.nn.functional as F
from esm.pretrained import ESM3_sm_open_v0  # 确保正确导入模型库

# 自定义数据集类，用于加载蛋白质序列
class ProteinDataset(Dataset):
    def __init__(self, fasta_file):
        self.proteins = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        return self.proteins[idx]

# 加载和准备模型
def load_and_prepare_model(device):
    model = esm3.ESM3_sm_open_v0(pretrained=True)
    model.to(device)
    # 设置微调参数
    for name, param in model.named_parameters():
        if 'output_heads.sequence_head.0' in name or 'output_heads.sequence_head.2' in name or 'output_heads.sequence_head.3' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

# 训练模型的函数
def train(model, loader, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for orig_seq in loader:
            # 转移到设备
            orig_seq = torch.tensor([orig_seq], dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(orig_seq)  # 确保模型可以接受并处理整数类型的输入
            
            # 计算损失
            loss = F.cross_entropy(outputs.logits, orig_seq)  # 确保损失计算适用于logits和标签
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader)}")

# 主程序
if __name__ == "__main__":
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_and_prepare_model(device)
    
    # 设置优化器
    optimizer = Adam(model.parameters(), lr=1e-4)

    # 加载数据
    fasta_file = "HA_train.fa"
    dataset = ProteinDataset(fasta_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 训练模型
    train(model, loader, optimizer, device)


```
