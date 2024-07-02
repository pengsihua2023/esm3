## 代码注释： /esm3/examples/local_client.py
```
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

# 导入所需的模块和类，用于蛋白质序列的生成、采样、解码、结构预测等功能。

def get_sample_protein() -> ESMProtein:
    protein = ProteinChain.from_rcsb("1utn")  # 从PDB数据库加载蛋白质结构数据。
    protein = ESMProtein.from_protein_chain(protein)  # 将ProteinChain对象转换为ESMProtein对象。
    protein.function_annotations = [
        # 添加功能注释，提供额外的蛋白质功能信息。
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    return protein  # 返回配置好的ESMProtein对象。

def main(client: ESM3InferenceClient):
    # 单步解码
    protein = get_sample_protein()  # 获取示例蛋白质对象。
    protein.function_annotations = None  # 移除功能注释。
    protein = client.encode(protein)  # 将蛋白质对象编码为张量。
    single_step_protein = client.forward_and_sample(
        protein,
        SamplingConfig(structure=SamplingTrackConfig(topk_logprobs=2)),
    )
    single_step_protein.protein_tensor.sequence = protein.sequence
    single_step_protein = client.decode(single_step_protein.protein_tensor)

    # 蛋白质折叠
    protein = get_sample_protein()
    sequence_length = len(protein.sequence)  # 获取序列长度。
    num_steps = int(sequence_length / 16)  # 计算折叠步骤数。
    protein.coordinates = None  # 移除坐标信息，准备进行折叠。
    protein.function_annotations = None  # 移除功能注释。
    protein.sasa = None  # 移除表面可接触面积信息。
    folded_protein = client.generate(
        protein,
        GenerationConfig(track="structure", schedule="cosine", num_steps=num_steps),
    )
    folded_protein.to_pdb("./sample_folded.pdb")  # 保存折叠后的蛋白质结构为PDB文件。

    # 逆向折叠
    protein = get_sample_protein()
    protein.sequence = None  # 移除序列信息，准备进行逆向折叠。
    protein.sasa = None
    protein.function_annotations = None
    inv_folded_protein = client.generate(
        protein,
        GenerationConfig(track="sequence", schedule="cosine", num_steps=num_steps),
    )
    inv_folded_protein.to_pdb("./sample_inv_folded.pdb")  # 保存逆向折叠结果。

    # 思维链模式：功能 -> 二级结构 -> 结构 -> 序列
    cot_protein = get_sample_protein()
    cot_protein.sequence = "_" * len(cot_protein.sequence)  # 用占位符初始化序列。
    cot_protein.coordinates = None
    cot_protein.sasa = None
    cot_protein_tensor = client.encode(cot_protein)
    for cot_track in ["secondary_structure", "structure", "sequence"]:
        cot_protein_tensor = client.generate(
            cot_protein_tensor,
            GenerationConfig(track=cot_track, schedule="cosine", num_steps=10),
        )
    cot_protein = client.decode(cot_protein_tensor)
    cot_protein.to_pdb("./sample_cot.pdb")  # 保存思维链模式的最终结果。

if __name__ == "__main__":
    main(ESM3.from_pretrained("esm3_sm_open_v1"))  # 使用预训练模型初始化客户端，并运行主函数。

```
### 总结
此代码示例使用ESM3模型进行蛋白质序列和结构的生成和解析。首先，通过PDB数据库获取蛋白质信息，然后进行单步解码、折叠、逆向折叠和思维链模式生成。这些步骤展示了如何使用深度学习模型预测和生成蛋白质的不同特性，包括功能注释、结构和序列信息。此外，代码还涉及对生成的蛋白质结构进行保存，便于后续的分析和研究。
