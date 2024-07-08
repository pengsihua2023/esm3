## 设置sequence-head可训练
如果希望仅将sequence_head的三层设置为可训练，并冻结模型中的其他所有层，您可以通过遍历模型的所有参数并根据其名字判断属于哪一部分来实现这一设置。下面是一段实现这一策略的示例代码：  
```
import torch
from esm.pretrained import ESM3_sm_open_v0

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ESM3_sm_open_v0(device=device)

# 遍历所有参数，决定哪些参数可训练
for name, param in model.named_parameters():
    # 只有属于sequence_head的参数设置为可训练
    if 'sequence_head' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 确认设置结果
for name, param in model.named_parameters():
    print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")


```
- 代码解释：
模型加载：首先加载预训练的模型，并确保它在正确的设备上（GPU或CPU）。  
- 参数设置：
使用named_parameters()方法获取模型的所有参数及其名称。  
通过检查参数名称中是否包含sequence_head来判断该参数是否属于sequence_head。如果是，将其requires_grad属性设置为True，表示这些参数在训练过程中需要更新。  
如果参数名称不包含sequence_head，则将requires_grad设置为False，冻结这些参数，这样它们在训练过程中不会更新。  
- 确认设置：最后，遍历所有参数并打印出其训练状态，以确保正确地设置了参数的可训练状态。  
这种方法确保了模型的sequence_head部分可以根据新的数据进行调整，而其他部分保持预训练时的状态不变，这有助于利用预训练模型的通用特征学习能力，同时针对特定任务进行优化。
