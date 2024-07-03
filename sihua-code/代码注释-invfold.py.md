## 代码注释： invfold.py
```
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数库

from esm.pretrained import (  # 从esm包导入预训练模型
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization.sequence_tokenizer import (  # 导入序列化工具
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain  # 导入处理蛋白质结构的类

if __name__ == "__main__":  # 检查是否直接运行此脚本
    tokenizer = EsmSequenceTokenizer()  # 初始化序列化工具
    encoder = ESM3_structure_encoder_v0("cuda")  # 将结构编码器加载到GPU
    model = ESM3_sm_open_v0("cuda")  # 将序列模型加载到GPU

    chain = ProteinChain.from_pdb("esm/data/1utn.pdb")  # 从PDB文件加载蛋白质链
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()  # 提取结构编码所需的输入
    coords = coords.cuda()  # 将坐标张量转移到GPU
    plddt = plddt.cuda()  # 将pLDDT张量转移到GPU
    residue_index = residue_index.cuda()  # 将残基索引张量转移到GPU
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)  # 使用编码器编码结构信息

    # 添加BOS/EOS填充
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)  # 对坐标进行填充
    plddt = F.pad(plddt, (1, 1), value=0)  # 对pLDDT进行填充
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)  # 对结构令牌进行填充
    structure_tokens[:, 0] = 4098  # 设置BOS标记
    structure_tokens[:, -1] = 4097  # 设置EOS标记

    output = model.forward(  # 将填充后的输入传递给模型
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )
    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)  # 将输出对数值转换为令牌索引
    sequence = tokenizer.decode(sequence_tokens[0])  # 解码令牌为氨基酸序列
    print(sequence)  # 打印解码的序列

```
