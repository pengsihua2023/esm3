1. [Embedding a sequence using ESM C](https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb)  
2. [Understanding the ESMProtein Class](https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/1_esmprotein.ipynb)  
3. [Advanced prompting with additional ESM3 tracks](https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/4_forge_generate.ipynb)  
## Embedding a sequence using ESM C代码注释
### Set up Forge client for ESM C
```
from getpass import getpass  # 导入 getpass 模块，用于安全地输入敏感信息（如 API Token）

# 使用 getpass 获取用户输入的 API Token，并在输入时不显示字符（增强安全性）
token = getpass("Token from Forge console: ")

# 用户在运行代码时，会看到提示："Token from Forge console: "
# 但输入的 Token 不会显示在屏幕上，以防泄露。

```
推荐使用 getpass()，特别是在共享或远程环境下，以避免泄露 Token。    
以下是对代码的详细注释：

```python
from esm.sdk import client  # 从 ESM SDK 导入 `client`，用于连接 ESM-C 服务器

# 通过 ESM SDK 创建一个客户端实例，连接到 EvolutionaryScale Forge API
model = client(
    model="esmc-300m-2024-12",  # 指定要使用的 ESM-C 预训练模型（300M 参数版本，2024 年 12 月版本）
    url="https://forge.evolutionaryscale.ai",  # 远程 API 服务器地址
    token=token  # 认证 Token，用于授权 API 访问
)
```

### **代码解析**
1. **导入 `client`**：  
   - `esm.sdk.client` 是 EvolutionaryScale Forge 平台提供的 API 客户端接口。
   - 它用于与远程服务器通信，进行推理计算。

2. **创建 `model` 客户端**：
   - 通过 `client()` 方法实例化一个 ESM-C 模型客户端。
   - 参数：
     - `"esmc-300m-2024-12"`：指定要使用的 **ESM-C 300M 参数**版本（2024 年 12 月更新）。
     - `url="https://forge.evolutionaryscale.ai"`：指定 API 服务器地址。
     - `token=token`：使用之前通过 `getpass()` 输入的 API Token 进行身份验证。

### **作用**
- 该代码连接到 EvolutionaryScale Forge 平台，并初始化一个远程推理模型。
- 该 `model` 可用于后续的蛋白质序列分析，如生成嵌入、结构预测等。
- **本地无需 GPU**，计算全部在远程服务器完成。

---

### **示例**
如果你想调用该 `model` 进行蛋白质序列推理，可以这样使用：
```python
from esm.sdk.api import ESMProtein

# 定义蛋白质序列
sequence = "MENSDNIMYQK"

# 处理蛋白质序列
protein = ESMProtein(sequence=sequence)
protein_tensor = model.encode(protein)  # 编码序列
embedding = model.logits(protein_tensor)  # 获取嵌入
print(embedding)
```

### **总结**
- 该代码**初始化 ESM-C 300M 远程推理模型**，用于蛋白质嵌入计算。
- 需要**API Token** 进行身份认证（通过 `getpass()` 获取）。
- **所有计算在远程服务器完成**，本地无需 GPU，仅需网络连接。
```
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器，用于并行处理多个任务
from typing import Sequence  # 导入 Sequence 类型，用于类型注解

# 从 ESM-3 SDK 导入所需的类
from esm.sdk.api import (
    ESM3InferenceClient,  # ESM-3 推理客户端，用于与 API 交互
    ESMProtein,  # 表示蛋白质序列的类
    ESMProteinError,  # 处理蛋白质相关错误的类
    LogitsConfig,  # 配置推理请求的类
    LogitsOutput,  # 存储推理结果的类
    ProteinType,  # 蛋白质类型的别名
)

# 配置 ESM-3 模型的嵌入参数
EMBEDDING_CONFIG = LogitsConfig(
    sequence=True,  # 处理整个蛋白质序列
    return_embeddings=True,  # 返回嵌入向量
    return_hidden_states=True  # 返回隐藏状态
)

def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    """
    使用 ESM-3 模型计算给定蛋白质序列的嵌入。
    
    参数:
    - model: ESM3InferenceClient - 预训练的 ESM-3 推理客户端
    - sequence: str - 输入的蛋白质序列
    
    返回:
    - LogitsOutput - 计算出的蛋白质嵌入
    """
    protein = ESMProtein(sequence=sequence)  # 创建 ESMProtein 实例
    protein_tensor = model.encode(protein)  # 对蛋白质进行编码
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)  # 获取嵌入和隐藏状态
    return output  # 返回嵌入结果

def batch_embed(
    model: ESM3InferenceClient, inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """
    并行处理一批蛋白质序列，获取它们的嵌入。

    由于 Forge API 支持自动批处理，我们在这里使用线程池进行并行计算，
    以加快多个蛋白质序列的处理速度。

    参数:
    - model: ESM3InferenceClient - 预训练的 ESM-3 推理客户端
    - inputs: Sequence[ProteinType] - 一批蛋白质序列（字符串）

    返回:
    - Sequence[LogitsOutput] - 计算出的蛋白质嵌入列表
    """
    with ThreadPoolExecutor() as executor:  # 使用线程池执行器进行并行处理
        # 提交所有任务到线程池，executor.submit() 以异步方式运行 embed_sequence()
        futures = [
            executor.submit(embed_sequence, model, protein) for protein in inputs
        ]
        
        results = []  # 用于存储所有的嵌入结果
        
        # 遍历每个任务的结果
        for future in futures:
            try:
                results.append(future.result())  # 获取任务执行结果并添加到列表
            except Exception as e:
                # 如果发生异常，返回一个 ESMProteinError 以进行错误处理
                results.append(ESMProteinError(500, str(e)))
    
    return results  # 返回所有蛋白质序列的嵌入结果


```
