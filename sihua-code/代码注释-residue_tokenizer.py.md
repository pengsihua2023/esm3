## residue_tokenizer.py  

```
from functools import cached_property  # 从functools模块导入cached_property，用于创建只计算一次且被存储的属性
from pathlib import Path  # 从pathlib模块导入Path，用于文件路径操作
from typing import Any  # 从typing模块导入Any，用于类型注解

import pandas as pd  # 导入pandas库，通常用于数据处理
import torch  # 导入PyTorch库，用于机器学习和张量操作
import torch.nn.functional as F  # 从torch.nn导入functional模块，用于低层神经网络操作

from esm.tokenization.tokenizer_base import EsmTokenizerBase  # 从esm.tokenization.tokenizer_base模块导入EsmTokenizerBase基类
from esm.utils.constants import esm3 as C  # 从esm.utils.constants模块导入常量esm3并命名为C

Sample = dict[str, Any]  # 定义Sample类型为字典，键为字符串，值为任意类型

class ResidueAnnotationsTokenizer(EsmTokenizerBase):  # 定义ResidueAnnotationsTokenizer类，继承自EsmTokenizerBase
    def __init__(  # 定义构造函数
        self,
        csv_path: str | None = None,  # csv文件路径参数，默认为None
        max_annotations: int = 16,  # 最大注释数参数，默认为16
    ):
        if csv_path is None:  # 如果csv_path为None
            csv_path = str(C.data_root() / C.RESID_CSV)  # 设置csv_path为默认路径
        self.csv_path = csv_path  # 实例变量保存csv文件路径
        self.max_annotations = max_annotations  # 实例变量保存最大注释数

    @cached_property
    def _description2label(self) -> dict[str, str]:  # 定义一个缓存属性，用于将描述映射到标签
        with Path(self.csv_path).open() as f:  # 打开csv文件
            df = pd.read_csv(f)  # 读取csv文件为DataFrame
        return dict(zip(df.label, df.label_clean))  # 返回标签到清洁标签的映射

    @cached_property
    def _labels(self) -> list[str]:  # 定义一个缓存属性，获取所有标签列表
        with Path(self.csv_path).open() as f:  # 打开csv文件
            df = pd.read_csv(f)  # 读取csv文件为DataFrame
        labels = (
            df.groupby("label_clean")["count"]  # 按清洁标签分组，并对计数求和
            .sum()
            .sort_values(ascending=False, kind="stable")  # 按计数降序排序
            .index.tolist()  # 获取排序后的标签列表
        )
        return labels  # 返回标签列表

    @cached_property
    def _label2id(self) -> dict[str, int]:  # 定义一个缓存属性，将标签映射到ID
        offset = len(self.special_tokens) + 1  # 计算偏移量，特殊标记数加1
        return {label: offset + i for i, label in enumerate(self._labels)}  # 创建并返回标签到ID的映射

    @cached_property
    def special_tokens(self) -> list[str]:  # 定义一个缓存属性，返回特殊标记列表
        return ["<pad>", "<motif>", "<unk>"]  # 返回特殊标记列表

    @cached_property
    def vocab(self):  # 定义一个缓存属性，返回完整词汇表
        annotation_tokens = [f"<ra:{id}>" for _, id in self._label2id.items()]  # 创建注释标记列表
        return self.special_tokens + ["<none>"] + annotation_tokens  # 返回包含特殊标记、"<none>"和注释标记的词汇表

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:  # 定义一个缓存属性，创建词汇到索引的映射
        return {token: token_id for token_id, token in enumerate(self.vocab)}  # 返回词汇到索引的映射

    @cached_property
    def vocabulary(self) -> list[str]:  # 定义一个缓存属性，返回完整词汇列表
        return [*self.special_tokens, "<none>", *self._labels]  # 返回包含特殊标记、"<none>"和所有标签的词汇列表

    # 以下是其他方法和属性定义，包括对序列的标记化、编码、解码等操作

```
### 总结
这段代码定义了一个ResidueAnnotationsTokenizer类，主要用于处理蛋白质残基的注释信息，通过读取CSV文件中的数据来建立标签和ID的映射。该类继承自EsmTokenizerBase，并且定义了多个缓存属性来存储从文件读取的数据，以提高数据访问效率。此外，该类还包括对蛋白质序列的标记化方法，能够将蛋白质残基的注释信息转换为特定的标记形式，适用于深度学习模型的输入。通过使用缓存属性和类型注解，代码的可读性和性能都得到了优化。  

