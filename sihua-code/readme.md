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
### 关于protein.sasa = None的说明
在代码中，表达式 protein.sasa = None 的作用是将 protein 对象的 sasa 属性设置为 None。这里的 sasa 代表的是 "Solvent Accessible Surface Area"（溶剂可及表面积），它是一个重要的生物物理属性，用来描述蛋白质中某个部分表面被溶剂（通常是水）所接触的面积。  

设置 sasa 为 None 可能是为了准备蛋白质进行某些处理或运算，比如蛋白质结构预测、折叠或逆向折叠，而在这些过程中原有的 sasa 值可能不再适用或需要重新计算。通过将其设置为 None，可以确保不会使用到旧的、不适用的或不准确的 sasa 数据。 

### 什么是“单步解码”（Single step decoding）？
在上下文中，“单步解码”（Single step decoding）通常指的是使用深度学习模型完成对单一输入的处理和输出生成的过程，只经过一次模型的前向传递和相关的采样步骤来获取结果。这种方式常见于处理自然语言处理任务或生物信息学的序列数据。

在您提供的代码段中，单步解码具体涉及以下几个步骤：

编码：首先，蛋白质数据通过 client.encode(protein) 被转换成模型能够处理的内部表示形式，通常是一种数值化的张量格式。

前向传递和采样：接下来，通过 client.forward_and_sample() 方法，模型对编码后的数据执行一次前向传递，基于模型当前的学习和数据特性，生成预测结果。此处，还包括了采样配置（通过 SamplingConfig），可能用于调整结果的多样性或精确度（如 topk_logprobs=2 指定了概率最高的两个结果用于采样）。

解码：最后，生成的结果（此时仍在内部张量格式）通过 client.decode() 转换回蛋白质的可理解格式，比如蛋白质序列。

这个过程在生物信息学中尤其重要，它允许快速地从模型中获取对某个蛋白质结构或功能的预测，而无需多步迭代或复杂的反馈循环，从而显著提高了处理速度和效率。在实际应用中，这种方法适用于需要快速决策或预测的场景。
### 代码中的“蛋白质折叠”的实现步骤是怎样的？
代码中的“蛋白质折叠”部分是使用了深度学习模型来预测蛋白质在三维空间中的结构。这一过程的具体实现步骤如下：

- 获取蛋白质样本：使用 get_sample_protein() 函数获取一个预设的蛋白质样本。这个样本是从PDB（蛋白质数据银行）中加载的，并已经加上了功能注释。

- 初始化参数：首先，计算蛋白质序列的长度并基于这个长度来决定模型运行的步数（num_steps）。这里使用的公式是序列长度除以16，这个比例因子可能根据具体的模型和任务需要进行调整。

- 清除不必要的数据：

protein.coordinates = None：移除蛋白质的坐标数据，因为我们需要模型来预测这些坐标。
protein.function_annotations = None：移除功能注释，使得模型专注于结构的生成。
protein.sasa = None：移除溶剂可接触表面积（SASA）信息，因为折叠过程需要重新计算这些值。
- 生成结构：使用 client.generate() 方法，传入蛋白质样本和生成配置（GenerationConfig）。这里指定了生成轨道为“structure”（结构），使用“cosine”调度方案，以及前面计算的步数。这个配置指示模型应如何逐步调整蛋白质结构，直到达到预期的三维构型。

- 保存结果：生成的蛋白质结构通过 folded_protein.to_pdb("./sample_folded.pdb") 保存为PDB文件格式，这是一种广泛使用的存储和描述蛋白质三维结构数据的格式。

整个蛋白质折叠过程是模拟蛋白质如何从一维序列折叠到其在生物体中实际的三维结构，这对于理解蛋白质的功能和进行药物设计等应用至关重要。通过深度学习模型，这种折叠过程可以在计算上进行模拟，提供对实验数据的补充或预测。  
## 如何理解：代码中的“思维链模式：功能 -> 二级结构 -> 结构 -> 序列”？
代码中的“思维链模式：功能 -> 二级结构 -> 结构 -> 序列”是一种先进的模型使用策略，它通过分阶段生成不同的生物学特征来逐步构建或优化蛋白质的模型。这种方法模仿了科学家在探索和理解蛋白质时的思考过程，从功能分析到结构的预测再到序列的优化，反映了一种从宏观到微观的理解和设计策略。

下面是“思维链模式”在代码中的具体实现步骤：

- 初始化蛋白质样本：
 使用 get_sample_protein() 函数获取蛋白质样本，并对其进行一些初始化设置，比如用占位符"_"填充序列，表示在此阶段不使用原有的序列信息。  
 清除蛋白质的坐标（cot_protein.coordinates = None）和表面可接触面积（cot_protein.sasa = None）信息，为接下来的结构和序列预测做准备。  

- 编码蛋白质：
通过 client.encode(cot_protein) 将初始化后的蛋白质对象转化为模型可以处理的张量表示形式。  

- 分阶段生成：
使用循环对每个特定的蛋白质特征（二级结构、结构、序列）进行逐步生成。每一步都调用 client.generate() 方法，并指定不同的生成轨道（track）：
1. 二级结构：首先生成蛋白质的二级结构，这是蛋白质结构的基本组成元素，如α-螺旋和β-折叠。
2. 结构：基于二级结构的信息，进一步生成整个蛋白质的三维空间结构。
3. 序列：最后，根据结构信息优化或预测蛋白质的氨基酸序列。

- 解码和保存结果：
1. 使用 client.decode(cot_protein_tensor) 将最终的张量结果解码回蛋白质格式。  
2. 将得到的蛋白质结构保存为PDB文件（cot_protein.to_pdb("./sample_cot.pdb")），以便后续分析或使用。  

这种“思维链模式”是一种系统性的方法，可以帮助模型更加全面和精确地理解和预测蛋白质的不同特性。通过这样的分阶段处理，可以逐步优化每个特征，每一步都依赖于前一步的输出，这样可以提高预测的准确性和效率。  

### 请详细解释代码中说的“ 逆向折叠”
代码中提到的“逆向折史”（Inverse Folding）是一种生物信息学技术，用于根据已知的蛋白质结构预测与之相匹配的氨基酸序列。这与传统的蛋白质折叠（预测蛋白质结构基于序列信息）相反，逆向折史关注于发现或设计能够折叠成特定三维形状的序列。这在蛋白质工程和合成生物学中尤为重要，因为它可以帮助设计新的蛋白质具有特定的功能和结构属性。

下面是代码中实现“逆向折史”的具体步骤：

1. **获取蛋白质样本**：
   - 使用 `get_sample_protein()` 函数获取一个蛋白质样本，这个样本已经含有三维结构信息（从PDB加载）。

2. **清除序列和其他数据**：
   - `protein.sequence = None`：移除蛋白质的序列信息。在逆向折史中，目标是基于结构推测出序列，因此初始序列被清除。
   - `protein.sasa = None`：移除溶剂可接触表面积信息，因为需要重新评估这些信息。
   - `protein.function_annotations = None`：移除功能注释，以便专注于结构和序列的生成。

3. **生成配置设置**：
   - 使用 `GenerationConfig` 类配置生成任务，设置生成轨道为 `"sequence"`，调度策略为 `"cosine"`，并指定迭代次数 `num_steps`。这些参数控制生成过程的详细程度和变化速度。

4. **逆向生成序列**：
   - 调用 `client.generate()` 方法，根据蛋白质的已知结构信息来生成匹配的序列。这一步是逆向折史的核心，它通过模型的学习能力，试图找到能够折叠到给定结构的氨基酸序列。

5. **保存结果**：
   - 将生成的序列（此时已绑定到蛋白质结构上）保存为PDB文件，文件名为 `"./sample_inv_folded.pdb"`。这允许进一步的生物学分析或实验验证。

通过逆向折史，研究者可以探索特定结构的设计空间，找到新的可能的序列，这些序列在自然界中可能未曾存在。这种技术对于定制蛋白质以应对特定的生物学或工业需求是非常有价值的。
