## 代码注释： invfold.py
### 总结：
主要功能：从蛋白质三维结构pdb文件开始， 
需要的模型：ESM3_sm_open_v0，ESM3_structure_encoder_v0  
- 这个脚本使用esm包中的预训练模型来编码并预测基于其三维结构的蛋白质序列。它展示了如何整合esm库中的多个模型和实用程序来处理复杂的生物分子数据，突出了深度学习在生物信息学中的应用。整个工作流包括数据加载、预处理、模型推断和后处理步骤，展示了计算生物学的端到端过程。 
- 在上述代码中，ESM3_structure_encoder_v0 是用来处理和编码蛋白质的三维结构信息的。如果没有使用这个结构编码器，你将无法从蛋白质的三维结构中提取特征，这些特征对于生成或预测蛋白质序列是至关重要的。 
- 具体来说，ESM3_structure_encoder_v0 负责从给定的三维坐标、pLDDT 值和残基索引中提取结构特征，并将这些特征转换为结构令牌（tokens）。这些结构令牌是后续模型 ESM3_sm_open_v0 用于生成序列的重要输入之一。  
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
  
