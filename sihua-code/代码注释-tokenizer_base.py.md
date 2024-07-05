##
```
from typing import Protocol, runtime_checkable  # 从typing模块导入Protocol和runtime_checkable

@runtime_checkable  # 这个装饰器允许在运行时检查类是否符合接口
class EsmTokenizerBase(Protocol):  # 定义一个名为EsmTokenizerBase的协议类
    def encode(self, *args, **kwargs):  # 定义一个encode方法，参数灵活，用于将文本转换成数字ID序列
        ...

    def decode(self, *args, **kwargs):  # 定义一个decode方法，参数灵活，用于将数字ID序列转换回文本
        ...

    @property
    def mask_token(self) -> str:  # 定义一个mask_token的属性，返回值为字符串，代表遮蔽标记
        ...

    @property
    def mask_token_id(self) -> int:  # 定义一个mask_token_id的属性，返回值为整数，代表遮蔽标记的ID
        ...

    @property
    def bos_token(self) -> str:  # 定义一个bos_token的属性，返回值为字符串，代表句子开始的标记
        ...

    @property
    def bos_token_id(self) -> int:  # 定义一个bos_token_id的属性，返回值为整数，代表句子开始标记的ID
        ...

    @property
    def eos_token(self) -> str:  # 定义一个eos_token的属性，返回值为字符串，代表句子结束的标记
        ...

    @property
    def eos_token_id(self) -> int:  # 定义一个eos_token_id的属性，返回值为整数，代表句子结束标记的ID
        ...

    @property
    def pad_token(self) -> str:  # 定义一个pad_token的属性，返回值为字符串，代表填充标记
        ...

    @property
    def pad_token_id(self) -> int:  # 定义一个pad_token_id的属性，返回值为整数，代表填充标记的ID
        ...


```
### ```python
from typing import Protocol, runtime_checkable  # 从typing模块导入Protocol和runtime_checkable

@runtime_checkable  # 这个装饰器允许在运行时检查类是否符合接口
class EsmTokenizerBase(Protocol):  # 定义一个名为EsmTokenizerBase的协议类
    def encode(self, *args, **kwargs):  # 定义一个encode方法，参数灵活，用于将文本转换成数字ID序列
        ...

    def decode(self, *args, **kwargs):  # 定义一个decode方法，参数灵活，用于将数字ID序列转换回文本
        ...

    @property
    def mask_token(self) -> str:  # 定义一个mask_token的属性，返回值为字符串，代表遮蔽标记
        ...

    @property
    def mask_token_id(self) -> int:  # 定义一个mask_token_id的属性，返回值为整数，代表遮蔽标记的ID
        ...

    @property
    def bos_token(self) -> str:  # 定义一个bos_token的属性，返回值为字符串，代表句子开始的标记
        ...

    @property
    def bos_token_id(self) -> int:  # 定义一个bos_token_id的属性，返回值为整数，代表句子开始标记的ID
        ...

    @property
    def eos_token(self) -> str:  # 定义一个eos_token的属性，返回值为字符串，代表句子结束的标记
        ...

    @property
    def eos_token_id(self) -> int:  # 定义一个eos_token_id的属性，返回值为整数，代表句子结束标记的ID
        ...

    @property
    def pad_token(self) -> str:  # 定义一个pad_token的属性，返回值为字符串，代表填充标记
        ...

    @property
    def pad_token_id(self) -> int:  # 定义一个pad_token_id的属性，返回值为整数，代表填充标记的ID
        ...
```

### 总结
这段代码定义了一个名为`EsmTokenizerBase`的协议类，使用了`runtime_checkable`装饰器来允许运行时的接口符合性检查。这个类作为一种基础的接口定义，规定了任何符合此协议的分词器类必须实现的方法和属性。这些包括：

- **编码（encode）和解码（decode）方法**：用于将文本转换为数值ID序列，以及将这些序列转换回文本。
- **特殊标记的属性**：定义了如遮蔽标记（mask_token）、开始标记（bos_token）、结束标记（eos_token）、填充标记（pad_token）及其对应的ID属性。

此协议的目的是确保任何实现此接口的类都将具有处理文本序列所需的基本功能和属性，便于统一管理和调用。
