## sasa_tokenizer.py

```
from functools import cached_property  # 导入cached_property，用于创建只计算一次且被缓存的属性

import torch  # 导入PyTorch库，用于张量操作

from esm.tokenization.tokenizer_base import EsmTokenizerBase  # 从esm模块导入基础分词器类
from esm.utils.constants import esm3 as C  # 导入常量配置，用于获取配置项

class SASADiscretizingTokenizer(EsmTokenizerBase):  # 定义SASA离散化分词器，继承自基础分词器
    """Tokenizer for Solvent Accessible Surface Area (SASA)."""
    def __init__(self, boundaries: list[float] = C.SASA_DISCRETIZATION_BOUNDARIES):
        self._boundaries = sorted(boundaries)  # 初始化时对给定的区间边界进行排序

    @cached_property
    def special_tokens(self) -> list[str]:
        return ["<pad>", "<motif>", "<unk>"]  # 定义特殊标记

    @cached_property
    def vocab(self) -> list[str]:
        """Discrete token vocabulary."""
        # 根据边界值创建词汇表，每个范围表示为"<low-high>"
        boundary_strs = ["0"] + [str(b) for b in self._boundaries] + ["inf"]
        range_tokens = [
            f"<{low}-{high}>"
            for low, high in zip(boundary_strs[:-1], boundary_strs[1:])
        ]
        return self.special_tokens + range_tokens  # 返回包含特殊标记的完整词汇表

    @cached_property
    def midpoints(self) -> list[float]:
        """Midpoints of the SASA token ranges."""
        # 计算各范围中点，用于解码
        boundaries = [0] + self._boundaries + [self._boundaries[-1] * 2]
        midpoint_tokens = [
            (float(high) + float(low)) / 2
            for low, high in zip(boundaries[:-1], boundaries[1:])
        ]
        midpoint_tokens = [float("nan"), float("nan"), float("nan")] + midpoint_tokens
        return midpoint_tokens

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        """Constructs token -> token id mapping."""
        return {word: i for i, word in enumerate(self.vocab)}  # 构建词汇到索引的映射

    def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Determines which positions are special tokens."""
        return tokens < len(self.special_tokens)  # 标记特殊标记的位置

    def encode(self, values: list[float | str], add_special_tokens: bool = True) -> torch.Tensor:
        """Encodes SASA values as discrete tokens."""
        ids = []
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # 添加开始标记
        for value in values:
            if isinstance(value, (float, int)):  # 数值根据边界划分到对应区间
                bucket = torch.bucketize(value, torch.tensor(self._boundaries))
                token_id = len(self.special_tokens) + bucket
            elif isinstance(value, str):  # 字符串直接查找索引
                token_id = self.vocab_to_index[value]
            else:
                raise TypeError(value)
            ids.append(token_id)
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # 添加结束标记
        return torch.tensor(ids, dtype=torch.int64)  # 返回编码后的张量

    def decode_float(self, encoded: torch.Tensor) -> list[float]:
        """Decodes SASA token ids into float values."""
        return [self.midpoints[token_id] for token_id in encoded]  # 解码为浮点数，使用区间中点

    def decode(self, encoded: torch.Tensor) -> str:
        """Decodes SASA token ids."""
        return ",".join(self.vocab[i] for i in encoded)  # 解码为字符串，用逗号分隔

    def decode_list(self, encoded: torch.Tensor) -> list[str]:
        """Decodes SASA token ids."""
        return [self.vocab[i] for i in encoded]  # 解码为字符串列表

    @property
    def mask_token(self) -> str:
        return "<pad>"  # 定义掩码标记

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]  # 返回掩码标记的索引

    @property
    def bos_token(self) -> str:
        return "<pad>"  # 定义开始标记

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]  # 返回开始标记的索引

    @property
    def eos_token(self) -> str:
        return "<pad>"  # 定义结束标记

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]  # 返回结束标记的索引

    @property
    def pad_token(self) -> str:
        return "<pad>"  # 定义填充标记

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]  # 返回填充标记的索引


```

### 总结
这个类是一个分词器，用于将溶剂可及表面积（SASA）的数值转换成离散的标记，这些标记表示数值所在的区间范围。通过这种方式，连续的数值被转换为离散的符号，便于在机器学习模型中使用。类中包括了对特殊标记的定义、词汇表的创建、以及编码和解码的功能，同时还提供了对特殊标记的快速检索。这些功能使得这个分词器非常适合在处理生物信息学和结构生物学数据时使用。  
