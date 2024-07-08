## fasta格式文件中蛋白质序列的封装
```
from Bio import SeqIO
from esm.sdk.api import ESMProtein

# 读取fasta文件
fasta_file = "HA_train.fa"

# 创建一个空的列表来存储 ESMProtein 对象
proteins = []

# 使用BioPython读取每个蛋白质序列
for record in SeqIO.parse(fasta_file, "fasta"):
    # 将序列封装成ESMProtein对象
    protein = ESMProtein(sequence=str(record.seq))
    # 将ESMProtein对象添加到列表中
    proteins.append(protein)

# 现在 proteins 列表包含了所有的 ESMProtein 对象


```
