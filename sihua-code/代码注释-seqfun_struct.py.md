## 代码注释：seqfun_struct.py
### 功能总结
该代码是用于蛋白质序列和功能的深度学习模型分析。首先，它通过序列和功能的编码器对输入的蛋白质序列进行编码，然后使用预训练的深度学习模型进行预测，包括蛋白质的结构和功能。在预测过程中，特定的序列位置被随机替换以模拟序列变异。最后，解码函数用于解释模型输出，包括结构的预测和功能注释，最终将结构信息输出为PDB格式文件，并打印功能预测结果。  
```
import random  # 导入random模块，用于生成随机数
import torch  # 导入PyTorch，用于张量计算
import torch.nn.functional as F  # 导入PyTorch的功能性API，例如softmax等操作

# 从esm库中导入预训练模型和分词器
from esm.pretrained import (
    ESM3_function_decoder_v0,  # 功能解码模型版本0
    ESM3_sm_open_v0,           # 结构模型开放版本0
    ESM3_structure_decoder_v0, # 结构解码模型版本0
)
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,  # 功能分词器，基于InterPro量化
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,  # 序列分词器
)
from esm.utils.constants.esm3 import (
    SEQUENCE_MASK_TOKEN,  # 序列掩码代号
)
from esm.utils.structure.protein_chain import ProteinChain  # 蛋白质链工具类
from esm.utils.types import FunctionAnnotation  # 功能注释类型

# 装饰器，该函数在调用时不会计算梯度
@torch.no_grad()
def main():
    tokenizer = EsmSequenceTokenizer()  # 初始化序列分词器
    function_tokenizer = EsmFunctionTokenizer()  # 初始化功能分词器

    model = ESM3_sm_open_v0("cuda")  # 加载模型到CUDA设备上

    # PDB 1UTN的序列
    sequence = "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    tokens = tokenizer.encode(sequence)  # 对序列进行编码

    # 计算需要替换的代号数量，排除首尾代号
    num_to_replace = int((len(tokens) - 2) * 0.75)

    # 随机选择要替换的索引，排除第一个和最后一个索引
    indices_to_replace = random.sample(range(1, len(tokens) - 1), num_to_replace)

    # 将选中的索引替换为序列掩码代号
    for idx in indices_to_replace:
        tokens[idx] = SEQUENCE_MASK_TOKEN
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)  # 转换为张量

    function_annotations = [
        # 函数注释：胰蛋白酶S1A, 胰蛋白酶家族
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    function_tokens = function_tokenizer.tokenize(function_annotations, len(sequence))  # 对功能注释进行分词
    function_tokens = function_tokenizer.encode(function_tokens)  # 编码功能分词结果

    function_tokens = function_tokens.cuda().unsqueeze(0)  # 转移到CUDA设备并增加一个维度
    sequence_tokens = sequence_tokens.cuda().unsqueeze(0)  # 同上

    output = model.forward(  # 调用模型的前向传播函数
        sequence_tokens=sequence_tokens, function_tokens=function_tokens
    )
    return sequence, output, sequence_tokens  # 返回序列、输出结果和序列代号


@torch.no_grad()  # 装饰器，该函数在调用时不会计算梯度
def decode(sequence, output, sequence_tokens):
    # 为了节省显存，这些在单独的函数中加载
    decoder = ESM3_structure_decoder_v0("cuda")  # 加载结构解码模型到CUDA设备
    function_decoder = ESM3_function_decoder_v0("cuda")  # 加载功能解码模型到CUDA设备
    function_tokenizer = EsmFunctionTokenizer()  # 初始化功能分词器

    structure_tokens = torch.argmax(output.structure_logits, dim=-1)  # 提取结构逻辑的最大值索引
    structure_tokens = (
        structure_tokens.where(sequence_tokens != 0, 4098)  # 替换序列起始代号为特殊代号
        .where(sequence_tokens != 2, 4097)  # 替换序列终止代号为特殊代号
        .where(sequence_tokens != 31, 4100)  # 替换链断代号为特殊代号
    )

    bb_coords = (
        decoder.decode(  # 解码结构代号
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )["bb_pred"]
        .detach()  # 分离计算图
        .cpu()  # 转移到CPU
    )

    chain = ProteinChain.from_backbone_atom_coordinates(
        bb_coords, sequence="X" + sequence + "X"
    )
    chain.infer_oxygen().to_pdb("hello.pdb")  # 生成PDB文件

    # 函数预测
    p_none_threshold = 0.05  # 阈值
    log_p = F.log_softmax(output.function_logits[:, 1:-1, :], dim=3).squeeze(0)  # 计算对数softmax并压缩维度

    # 选择没有预测功能的位置
    log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
    p_none = torch.exp(log_p_nones).mean(dim=1)  # 计算平均"无功能"的概率
    where_none = p_none > p_none_threshold  # 判断无功能的位置

    log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
    function_token_ids = torch.argmax(log_p, dim=2)
    function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]

    predicted_function = function_decoder.decode(  # 解码预测的功能
        function_token_ids,
        tokenizer=function_tokenizer,
        annotation_threshold=0.1,
        annotation_min_length=5,
        annotation_gap_merge_max=3,
    )

    print("function prediction:")  # 打印功能预测结果
    print(predicted_function["interpro_preds"].nonzero())
    print(predicted_function["function_keywords"])


if __name__ == "__main__":
    sequence, output, sequence_tokens = main()  # 运行主函数
    torch.cuda.empty_cache()  # 清空CUDA缓存
    decode(sequence, output, sequence_tokens)  # 运行解码函数

```
  
