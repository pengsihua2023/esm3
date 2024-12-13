## raw_forwards-代码分析
提供的Python脚本 `raw_forwards.py` 利用Facebook AI的进化规模建模（ESM）库执行先进的蛋白质序列和结构预测任务。以下是对该代码的全面分析，详细说明其组件、功能和整体工作流程。

## **1. 导入语句**

```python
import random

import torch
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation
```

### **解释：**
- **标准库：**
  - `random`：用于随机操作，特别是用于掩蔽标记。
  - `torch` 和 `torch.nn.functional`：核心的PyTorch库，用于张量操作和神经网络功能。

- **ESM库模块：**
  - **预训练模型：**
    - `ESM3_function_decoder_v0`
    - `ESM3_sm_open_v0`
    - `ESM3_structure_decoder_v0`
    - `ESM3_structure_encoder_v0`
  
  - **标记化：**
    - `get_model_tokenizers`：获取序列和功能的标记器。
    - `InterProQuantizedTokenizer`（别名 `EsmFunctionTokenizer`）：专门用于功能注释的标记器。
    - `EsmSequenceTokenizer`：用于蛋白质序列的标记器。
  
  - **实用工具：**
    - `ProteinChain`：用于处理蛋白质链，特别是从PDB文件加载。
    - `FunctionAnnotation`：用于注释蛋白质功能的数据结构。

## **2. 函数：`inverse_folding_example()`**

```python
@torch.no_grad()
def inverse_folding_example():
    tokenizer = EsmSequenceTokenizer()
    encoder = ESM3_structure_encoder_v0("cuda")
    model = ESM3_sm_open_v0("cuda")

    chain = ProteinChain.from_rcsb("1utn", "A")
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cuda()
    plddt = plddt.cuda()
    residue_index = residue_index.cuda()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )

    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
    sequence = tokenizer.decode(sequence_tokens[0])
    print(sequence)
```

### **目的：**
**逆折叠**——从给定的蛋白质结构预测氨基酸序列。

### **工作流程：**
1. **初始化：**
   - **标记器：** 初始化一个序列标记器，用于在序列和标记ID之间转换。
   - **编码器与模型：** 将结构编码器和主要的ESM3模型 (`ESM3_sm_open_v0`) 加载到GPU上。

2. **加载蛋白质结构：**
   - **ProteinChain：** 从蛋白质数据库（PDB）条目 `1utn` 的链 "A" 加载蛋白质链。
   - **编码输入：** 将蛋白质链转换为编码器所需的结构输入：
     - `coords`：蛋白质主链原子的三维坐标。
     - `plddt`：每个残基的置信度评分。
     - `residue_index`：残基的索引。

3. **填充和特殊标记：**
   - **填充：** 对坐标、pLDDT评分和结构标记进行填充，以满足模型的要求。
   - **特殊标记：** 使用特定的标记ID（`4098` 代表序列开始（BOS），`4097` 代表序列结束（EOS））设置序列的开始和结束标记。

4. **模型推理：**
   - **前向传播：** 将填充后的坐标、pLDDT评分和结构标记输入模型，以获得序列预测的logits。
   - **序列预测：** 使用 `argmax` 选择每个位置最可能的标记，解码标记为氨基酸序列并打印出来。

### **结果：**
该函数通过利用ESM3模型的逆折叠能力，从提供的蛋白质结构（PDB `1utn` 链A）重建氨基酸序列，并将其打印出来。

## **3. 函数：`conditioned_prediction_example()`**

```python
@torch.no_grad()
def conditioned_prediction_example():
    tokenizers = get_model_tokenizers()

    model = ESM3_sm_open_v0("cuda")

    # PDB 1UTN
    sequence = "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    tokens = tokenizers.sequence.encode(sequence)

    # Calculate the number of tokens to replace, excluding the first and last token
    num_to_replace = int((len(tokens) - 2) * 0.75)

    # Randomly select indices to replace, excluding the first and last index
    indices_to_replace = random.sample(range(1, len(tokens) - 1), num_to_replace)

    # Replace selected indices with 32
    assert tokenizers.sequence.mask_token_id is not None
    for idx in indices_to_replace:
        tokens[idx] = tokenizers.sequence.mask_token_id
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)

    function_annotations = [
        # Peptidase S1A, chymotrypsin family
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    function_tokens = tokenizers.function.tokenize(function_annotations, len(sequence))
    function_tokens = tokenizers.function.encode(function_tokens)

    function_tokens = function_tokens.cuda().unsqueeze(0)
    sequence_tokens = sequence_tokens.cuda().unsqueeze(0)

    output = model.forward(
        sequence_tokens=sequence_tokens, function_tokens=function_tokens
    )
    return sequence, output, sequence_tokens
```

### **目的：**
**条件预测**——在特定功能注释的条件下，预测被掩蔽的蛋白质序列部分。

### **工作流程：**
1. **初始化：**
   - **标记器：** 获取序列和功能的标记器。
   - **模型：** 将ESM3模型 (`ESM3_sm_open_v0`) 加载到GPU上。

2. **序列准备：**
   - **序列定义：** 定义一个特定的蛋白质序列（可能对应于PDB条目 `1utn`）。
   - **编码：** 使用序列标记器将氨基酸序列转换为标记ID。

3. **随机掩蔽：**
   - **确定掩蔽数量：** 计算需要掩蔽的标记数量（不包括第一个和最后一个标记的75%）。
   - **选择索引：** 随机选择要掩蔽的标记位置，确保第一个和最后一个标记不被掩蔽。
   - **应用掩蔽：** 将选定的标记替换为掩蔽标记ID（`32`），这是由标记器定义的。

4. **功能注释：**
   - **注释定义：** 在序列中定义两个功能区域：
     - 第100到114位的 `"peptidase"`（蛋白酶）。
     - 第190到202位的 `"chymotrypsin"`（胰凝乳蛋白酶）。
   - **标记化：** 使用功能标记器将这些注释转换为标记ID。

5. **为模型准备输入：**
   - **张量转换：** 将序列和功能标记转换为PyTorch张量。
   - **设备分配：** 将张量移动到GPU上，并添加批次维度。

6. **模型推理：**
   - **前向传播：** 将掩蔽后的序列标记和功能标记输入模型，以获得预测结果。
   - **输出：** 返回原始序列、模型输出和掩蔽后的序列标记，用于进一步解码。

### **结果：**
该函数准备了一个被掩蔽的蛋白质序列，结合特定的功能注释，利用ESM3模型预测被掩蔽的部分。这展示了模型基于功能信息执行上下文感知序列补全的能力。

## **4. 函数：`decode(sequence, output, sequence_tokens)`**

```python
@torch.no_grad()
def decode(sequence, output, sequence_tokens):
    # To save on VRAM, we load these in separate functions
    decoder = ESM3_structure_decoder_v0("cuda")
    function_decoder = ESM3_function_decoder_v0("cuda")
    function_tokenizer = EsmFunctionTokenizer()

    # Generally not recommended to just argmax the logits, decode iteratively!
    # For quick demonstration only:
    structure_tokens = torch.argmax(output.structure_logits, dim=-1)
    structure_tokens = (
        structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
        .where(sequence_tokens != 2, 4097)  # EOS
        .where(sequence_tokens != 31, 4100)  # Chainbreak
    )

    bb_coords = (
        decoder.decode(
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )["bb_pred"]
        .detach()
        .cpu()
    )

    chain = ProteinChain.from_backbone_atom_coordinates(
        bb_coords, sequence="X" + sequence + "X"
    )
    chain.infer_oxygen().to_pdb("hello.pdb")

    # Function prediction
    p_none_threshold = 0.05
    log_p = F.log_softmax(output.function_logits[:, 1:-1, :], dim=3).squeeze(0)

    # Choose which positions have no predicted function.
    log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
    p_none = torch.exp(log_p_nones).mean(dim=1)  # "Ensemble of <none> predictions"
    where_none = p_none > p_none_threshold  # (length,)

    log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
    function_token_ids = torch.argmax(log_p, dim=2)
    function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]

    predicted_function = function_decoder.decode(
        function_token_ids,
        tokenizer=function_tokenizer,
        annotation_threshold=0.1,
        annotation_min_length=5,
        annotation_gap_merge_max=3,
    )

    print("function prediction:")
    print(predicted_function["interpro_preds"].nonzero())
    print(predicted_function["function_keywords"])
```

### **目的：**
**解码输出**——将模型的预测结果转化为有意义的蛋白质结构和功能注释。

### **工作流程：**
1. **初始化：**
   - **解码器：**
     - `ESM3_structure_decoder_v0`：将结构标记解码为主链坐标。
     - `ESM3_function_decoder_v0`：将功能标记解码为功能注释。
   - **功能标记器：** 初始化用于处理功能相关标记的功能标记器。

2. **结构解码：**
   - **结构标记：** 使用 `argmax` 从模型的 `structure_logits` 中选择每个位置最可能的结构标记。
   - **特殊标记调整：**
     - **BOS（4098）：** 替换 `sequence_tokens == 0` 的标记。
     - **EOS（4097）：** 替换 `sequence_tokens == 2` 的标记。
     - **链断裂（4100）：** 替换 `sequence_tokens == 31` 的标记。
   
   - **主链坐标：**
     - **解码：** 将调整后的结构标记输入结构解码器，获取主链坐标（`bb_pred`）。
     - **转换：** 将主链坐标转换为 `ProteinChain` 对象，在序列两端添加填充残基（`"X"`）。
     - **推断与保存：** 推断氧原子并将结构保存为名为 `hello.pdb` 的PDB文件。

3. **功能预测解码：**
   - **阈值设定：**
     - **概率计算：** 对 `function_logits`（不包括BOS和EOS标记）应用 `log_softmax`，获得对数概率。
     - **"<none>" 概率：** 提取对应 `"<none>"` 功能的对数概率，并在相关维度上计算平均概率。
     - **掩蔽：** 确定 `"<none>"` 概率超过阈值（`0.05`）的位置，表示这些位置没有功能注释。
   
   - **功能标记选择：**
     - **无效化 "<none>"：** 在不满足条件的位置将 `"<none>"` 的对数概率设为负无穷。
     - **Argmax 选择：** 在每个位置选择最可能的功能标记。
     - **应用 "<none>"：** 对于超过阈值的位置，分配 `"<none>"`。
   
   - **解码功能注释：**
     - **功能解码器：** 将选择的功能标记ID转换为有意义的功能注释，应用注释置信度阈值、最小长度和间隙合并。
     - **输出：** 打印非零的InterPro预测和功能关键词，有效总结预测的功能区域。

### **结果：**
该函数将模型的原始输出转化为：
- 作为 `hello.pdb` 保存的三维蛋白质结构。
- 预测的功能注释，突出显示具有特定功能的区域或指示没有功能（`"<none>"`）的位置。

## **5. 主执行块**

```python
if __name__ == "__main__":
    inverse_folding_example()

    sequence, output, sequence_tokens = conditioned_prediction_example()
    torch.cuda.empty_cache()
    # And then decode from tokenized representation to outputs:
    decode(sequence, output, sequence_tokens)
```

### **工作流程：**
1. **逆折叠：**
   - **函数调用：** 执行 `inverse_folding_example()` 以预测并打印从给定蛋白质结构重建的氨基酸序列。

2. **条件预测：**
   - **函数调用：** 执行 `conditioned_prediction_example()` 以执行在功能注释条件下的掩蔽序列预测。
   - **缓存管理：** 清空CUDA缓存以有效管理GPU内存。

3. **解码：**
   - **函数调用：** 执行 `decode(sequence, output, sequence_tokens)`，将模型的预测结果转化为三维结构和功能注释。

### **整体工作流程：**
该脚本依次执行逆折叠和条件预测，随后解码结果以生成蛋白质的结构和功能见解。

## **6. 功能总结**

1. **逆折叠 (`inverse_folding_example`)：**
   - **输入：** 来自PDB的蛋白质结构（`1utn` 链A）。
   - **过程：** 对结构进行编码，添加填充和特殊标记，使用ESM3模型预测对应的氨基酸序列。
   - **输出：** 打印重建的氨基酸序列。

2. **条件预测 (`conditioned_prediction_example`)：**
   - **输入：** 一个特定的蛋白质序列，其中75%的标记被掩蔽，并具有预定义的功能注释（`"peptidase"` 和 `"chymotrypsin"` 区域）。
   - **过程：** 编码掩蔽的序列和功能注释，使用ESM3模型预测掩蔽的标记。
   - **输出：** 返回原始序列、模型输出和掩蔽后的序列标记，用于进一步解码。

3. **解码 (`decode`)：**
   - **输入：** 来自条件预测的输出。
   - **过程：**
     - 将结构logits解码为主链坐标，并将结构保存为PDB文件（`hello.pdb`）。
     - 处理功能logits以确定功能注释，应用阈值过滤掉没有显著功能预测的位置。
   - **输出：** 保存三维结构并打印预测的功能注释。

## **7. 潜在应用**

- **蛋白质设计：** 从期望的结构逆向设计蛋白质序列。
- **功能注释：** 基于序列和结构数据预测蛋白质中的功能区域。
- **结构生物学：** 计算生成和分析蛋白质结构。
- **药物发现：** 识别可供治疗药物靶向的功能位点。

## **8. 考虑因素和建议**

- **解码策略：** 当前的解码方法使用 `argmax`，可能无法捕捉序列或功能的完整分布。实施更复杂的解码方法（如束搜索）可能会提高预测质量。
  
- **掩蔽策略：** 随机掩蔽75%的标记比例相当高。根据应用需求，调整掩蔽比例或模式可能会带来不同的见解。
  
- **错误处理：** 引入错误检查（如确保成功下载PDB文件、处理无效注释）可以提高脚本的健壮性。
  
- **性能优化：** 更有效地管理GPU内存，特别是在处理较大蛋白质或多链时，可以提高可扩展性。

## **结论**

`raw_forwards.py` 脚本展示了ESM3模型在蛋白质科学中的高级应用，演示了逆折叠和条件序列预测的能力。通过结合结构数据和功能注释，脚本促进了对蛋白质序列和结构的全面分析，为计算生物学和生物信息学的创新研究与发展铺平了道路。
