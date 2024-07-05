## 序列分词器
```
from tokenizers import Tokenizer                      # 从tokenizers库导入Tokenizer类
from tokenizers.models import BPE                     # 从tokenizers.models导入BPE模型
from tokenizers.processors import TemplateProcessing  # 从tokenizers.processors导入TemplateProcessing类
from transformers import PreTrainedTokenizerFast      # 从transformers库导入PreTrainedTokenizerFast类

from esm.tokenization.tokenizer_base import EsmTokenizerBase  # 从esm.tokenization.tokenizer_base导入EsmTokenizerBase基类
from esm.utils.constants import esm3 as C                     # 从esm.utils.constants导入esm3常数并重命名为C

class EsmSequenceTokenizer(PreTrainedTokenizerFast, EsmTokenizerBase):  # 定义EsmSequenceTokenizer类，继承自PreTrainedTokenizerFast和EsmTokenizerBase
    """
    构造一个ESM分词器。
    """

    model_input_names = ["sequence_tokens", "attention_mask"]  # 定义模型输入名

    def __init__(  # 定义初始化方法
        self,
        unk_token="<unk>",    # 定义未知词标记
        cls_token="<cls>",    # 定义开始句子的标记
        pad_token="<pad>",    # 定义填充标记
        mask_token="<mask>",  # 定义遮蔽标记
        eos_token="<eos>",    # 定义结束句子的标记
        chainbreak_token="|", # 定义链断裂标记
        **kwargs,
    ):
        all_tokens = C.SEQUENCE_VOCAB                  # 从常量中获取所有词汇
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}  # 创建从词汇到ID的映射

        # 一个字符级的分词器等同于一个没有合并规则的BPE分词器
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)  # 创建BPE模型实例
        tokenizer = Tokenizer(bpe)  # 创建Tokenizer实例
        special_tokens = [cls_token, pad_token, mask_token, eos_token, chainbreak_token]  # 定义特殊标记
        additional_special_tokens = [chainbreak_token]  # 定义额外的特殊标记

        tokenizer.add_special_tokens(  # 向分词器中添加特殊标记
            special_tokens,
        )

        # 在这里配置当调用tokenizer(text, add_special_tokens=True)时自动添加特殊标记。这里也可以配置如何合并两个序列。
        tokenizer.post_processor = TemplateProcessing(  # 设置分词器的后处理器
            single="<cls> $A <eos>",  # 设置单个句子的处理模板
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(  # 初始化基类
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # 这些是陷阱，我们从未在任何地方使用`bos`标记，所以我们在这里重写它。
    @property
    def bos_token(self):
        return self.cls_token  # 定义bos_token属性，返回cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id  # 定义bos_token_id属性，返回cls_token的ID


```
### 总结
这段代码定义了一个名为EsmSequenceTokenizer的类，用于构建基于Byte-Pair Encoding (BPE) 方法的分词器。该分词器是针对ESM模型定制的，支持特殊标记的自动添加，并处理字符级的分词。此外，代码中定义了一些额外的属性和方法，用于处理特殊标记和优化分词过程。通过继承PreTrainedTokenizerFast和EsmTokenizerBase，它具备了处理预训练模型分词所需的多数功能。
