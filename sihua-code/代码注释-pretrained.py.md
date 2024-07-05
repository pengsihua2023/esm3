## pretrained.py
### 功能总结
这段代码主要定义了一系列函数来加载不同配置的ESM3模型，并将它们注册到一个本地模型注册表中。这些模型分别对应不同的功能解码器和结构编码解码器。模型都设置为评估模式，并加载相应的预训练权重。用户可以通过指定模型名称和设备（CPU或GPU）来加载相应的模型。此外，还提供了一个函数来向模型注册表中注册新的模型版本，增加模型的可扩展性和灵活性。这使得模型加载过程更为简化和标准化，便于进行局部或线上推理。

```
from typing import Callable  # 导入Callable，用于类型注解，表示可调用对象

import torch  # 导入torch库，用于深度学习操作
import torch.nn as nn  # 导入torch.nn，用于构建神经网络

# 从esm模块导入不同的模型和组件
from esm.models.esm3 import ESM3
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.utils.constants.esm3 import data_root  # 导入配置项，用于获取数据根目录
from esm.utils.constants.models import (  # 导入模型配置常量
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
)

ModelBuilder = Callable[[torch.device | str], nn.Module]  # 定义ModelBuilder类型，为构建模型的函数

def ESM3_sm_open_v0(device: torch.device | str = "cpu"):  # 定义函数加载特定版本的ESM3模型
    model = (
        ESM3(
            d_model=1536,  # 模型维度
            n_heads=24,  # 注意力头数
            v_heads=256,  # 变体头数
            n_layers=48,  # 层数
            structure_encoder_name=ESM3_STRUCTURE_ENCODER_V0,  # 结构编码器版本
            structure_decoder_name=ESM3_STRUCTURE_DECODER_V0,  # 结构解码器版本
            function_decoder_name=ESM3_FUNCTION_DECODER_V0,  # 功能解码器版本
        )
        .to(device)  # 将模型移动到指定的设备（CPU或GPU）
        .eval()  # 设置为评估模式
    )
    state_dict = torch.load(
        data_root() / "data/weights/esm3_sm_open_v1.pth", map_location=device  # 加载模型权重
    )
    model.load_state_dict(state_dict)  # 应用模型权重
    return model  # 返回模型

def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):  # 加载结构编码器模型
    model = (
        StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        )
        .to(device)
        .eval()
    )
    state_dict = torch.load(
        data_root() / "data/weights/esm3_structure_encoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):  # 加载结构解码器模型
    model = (
        StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).to(device).eval()
    )
    state_dict = torch.load(
        data_root() / "data/weights/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):  # 加载功能解码器模型
    model = FunctionTokenDecoder().to(device).eval()
    state_dict = torch.load(
        data_root() / "data/weights/esm3_function_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {  # 定义本地模型注册表
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
}

def load_local_model(model_name: str, device: torch.device | str = "cpu") -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:  # 检查模型是否已注册
        raise ValueError(f"Model {model_name} not found in local model registry.")
    return LOCAL_MODEL_REGISTRY[model_name](device)  # 加载模型

def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:  # 注册新模型
    LOCAL_MODEL_REGISTRY[model_name] = model_builder  # 将模型添加到注册表

```

