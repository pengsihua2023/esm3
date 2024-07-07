## 第一步微调：领域适应
```
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from esm.pretrained import esm3_sm_open_v0  # 确保从正确的库导入
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from Bio import SeqIO

# 定义一个数据集类，用于处理fasta格式的蛋白质序列
class ProteinSequenceDataset(Dataset):
    def __init__(self, fasta_path):
        self.sequences = []  # 存储序列的列表
        self.tokenizer = EsmSequenceTokenizer()  # 初始化序列分词器
        # 读取fasta文件中的每条记录，并对序列进行编码
        for record in SeqIO.parse(fasta_path, "fasta"):
            tokens = self.tokenizer.encode(record.seq)
            self.sequences.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.sequences)  # 返回数据集中的序列数
    
    def __getitem__(self, idx):
        return self.sequences[idx]  # 获取指定索引的序列

# 微调模型的函数
def finetune_model(model, train_dataset, val_dataset, device):
    model.train()  # 将模型设置为训练模式
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()  # 确保模型处于训练模式
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        
        model.eval()  # 设置模型为评估模式
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = outputs['loss']
                total_val_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_train_loss / len(train_loader)}, Validation Loss: {total_val_loss / len(val_loader)}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
    model = esm3_sm_open_v0(pretrained=True).to(device)  # 加载预训练模型
    train_dataset = ProteinSequenceDataset("traindata.fasta")  # 创建训练数据集实例
    val_dataset = ProteinSequenceDataset("valdata.fasta")  # 创建验证数据集实例
    finetune_model(model, train_dataset, val_dataset, device)  # 调用微调函数
    torch.save(model.state_dict(), "ESM3_sm_open_v0-Influenza-A.pth")  # 保存训练后的模型

if __name__ == "__main__":
    main()  # 运行主函数

```
