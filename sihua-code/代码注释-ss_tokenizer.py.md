## ss_tokenizer.py  

```
from functools import cached_property  # 从functools模块导入cached_property，用于缓存属性值
from typing import Sequence  # 从typing模块导入Sequence，用于类型注解

import torch  # 导入torch模块，用于处理张量

from esm.tokenization.tokenizer_base import EsmTokenizerBase  # 从esm.tokenization.tokenizer_base模块导入EsmTokenizerBase基类
from esm.utils.constants import esm3 as C  # 从esm.utils.constants模块导入常量esm3并命名为C

class SecondaryStructureTokenizer(EsmTokenizerBase):  # 定义一个名为SecondaryStructureTokenizer的类，继承自EsmTokenizerBase
    """Tokenizer for secondary structure strings."""  # 类的文档字符串说明这是一个用于二级结构字符串的分词器

    def __init__(self, kind: str = "ss8"):  # 初始化方法，带有一个默认参数kind
        assert kind in ("ss8", "ss3")  # 断言kind必须是"ss8"或"ss3"之一
        self.kind = kind  # 将参数kind保存到实例变量kind中

    @property
    def special_tokens(self) -> list[str]:  # 定义一个属性special_tokens，返回特殊令牌的列表
        return ["<pad>", "<motif>", "<unk>"]  # 返回特殊令牌的列表

    @cached_property
    def vocab(self):  # 定义一个名为vocab的缓存属性
        """Tokenzier vocabulary list."""  # 文档字符串说明这是词汇表列表
        match self.kind:  # 根据kind的值选择不同的选项
            case "ss8":  # 如果kind是"ss8"
                nonspecial_tokens = list(C.SSE_8CLASS_VOCAB)  # 从常量C中获取8类二级结构的词汇表
            case "ss3":  # 如果kind是"ss3"
                nonspecial_tokens = list(C.SSE_3CLASS_VOCAB)  # 从常量C中获取3类二级结构的词汇表
            case _:  # 如果都不是，即kind值不正确
                raise ValueError(self.kind)  # 抛出一个值错误异常
        return [*self.special_tokens, *nonspecial_tokens]  # 返回包括特殊令牌和非特殊令牌的完整词汇表

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:  # 定义一个名为vocab_to_index的缓存属性，用于建立词汇到索引的映射
        """Constructs token -> token id mapping."""  # 文档字符串说明这是构建令牌到令牌ID的映射
        return {word: i for i, word in enumerate(self.vocab)}  # 使用枚举函数和字典推导式构建映射

    def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:  # 定义一个方法，用于获取特殊令牌的掩码
        """Determines which positions are special tokens.
        Args:
            tokens: <int>[length]
        Returns:
            <bool>[length] tensor, true where special tokens are located in the input.
        """
        return tokens < len(self.special_tokens)  # 返回一个布尔型张量，标记输入中特殊令牌的位置

    def encode(
        self, sequence: str | Sequence[str], add_special_tokens: bool = True  # 定义一个方法，用于编码二级结构字符串
    ) -> torch.Tensor:
        """Encode secondary structure string
        Args:
            string: secondary structure string e.g. "GHHIT", or as token listk.
        Returns:
            <int>[sequence_length] token ids representing. Will add <cls>/<eos>.
        """
        ids = []  # 初始化一个空列表用于存储ID
        if add_special_tokens:  # 如果需要添加特殊令牌
            ids.append(self.vocab_to_index["<pad>"])  # 将<pad>令牌的ID添加到列表作为开始令牌
        for char in sequence:  # 遍历输入的字符串
            ids.append(self.vocab_to_index[char])  # 将每个字符对应的ID添加到列表
        if add_special_tokens:  # 如果需要添加特殊令牌
            ids.append(self.vocab_to_index["<pad>"])  # 将<pad>令牌的ID添加到列表作为结束令牌
        return torch.tensor(ids, dtype=torch.int64)  # 将列表转换成整数型张量并返回

    def decode(self, encoded: torch.Tensor) -> str:  # 定义一个方法，用于解码令牌ID为二级结构字符串
        """Decodes token ids into secondary structure string.
        Args:
            encoded: <int>[length] token id array.
        Returns
            Decoded secondary structure string.
        """
        return "".join(self.vocab[i] for i in encoded)  # 使用列表推导式和字符串连接方法将令牌ID转换回字符串并返回

    @property
    def mask_token(self) -> str:  # 定义一个属性mask_token，返回遮蔽令牌
        return "<pad>"  # 返回<pad>作为遮蔽令牌

    @property
    def mask_token_id(self) -> int:  # 定义一个属性mask_token_id，返回遮蔽令牌的ID
        return self.vocab_to_index[self.mask_token]  # 返回遮蔽令牌的ID

    @property
    def bos_token(self) -> str:  # 定义一个属性bos_token，返回开始令牌
        return "<pad>"  # 返回<pad>作为开始令牌

    @property
    def bos_token_id(self) -> int:  # 定义一个属性bos_token_id，返回开始令牌的ID
        return self.vocab_to_index[self.bos_token]  # 返回开始令牌的ID

    @property
    def eos_token(self) -> str:  # 定义一个属性eos_token，返回结束令牌
        return "<pad>"  # 返回<pad>作为结束令牌

    @property
    def eos_token_id(self) -> int:  # 定义一个属性eos_token_id，返回结束令牌的ID
        return self.vocab_to_index[self.eos_token]  # 返回结束令牌的ID

    @property
    def pad_token(self) -> str:  # 定
        return "<pad>"  # 返回<pad>作为填充令牌

    @property
    def pad_token_id(self) -> int:  # 定义一个属性pad_token_id，返回填充令牌的ID
        return self.vocab_to_index[self.pad_token]  # 返回填充令牌的ID

```
### 总结
这段代码定义了一个SecondaryStructureTokenizer类，专用于编码和解码蛋白质的二级结构字符串。它继承自EsmTokenizerBase，提供了自定义的编码和解码功能，以及对特殊令牌的处理。类中包含了处理不同种类的二级结构（如ss8和ss3）的逻辑，利用词汇表和令牌到索引的映射来转换字符串与令牌ID之间的关系。此外，该类使用<pad>令牌同时作为遮蔽、开始和结束标记，这在某些NLP任务中是常见的简化做法。
