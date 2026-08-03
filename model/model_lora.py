'''
Author: Ethan
Date: 2026-07-28 21:17:54
LastEditors: Ethan
LastEditTime: 2026-08-03 15:19:21
Description: 
FilePath: /model/model_lora.py
'''
import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        # rank -> 控制矩阵的秩
        # 先执行 A线性层 -> 输出执行B线性层
        return self.B(self.A(x))

"""LoRA 的研究和实践一致表明：在 attention 层加低秩适配性价比最高，FFN 层额外加 LoRA 收益递减。 Q、K、V、O 控制了"注意谁"，是语言理解最上游的决策点；FFN 只是做非线性变换，对微调（尤其是 SFT/GRPO 这种指令对齐）影响偏小。
- 它只对"输入维度 = 输出维度"的 Linear 层做 LoRA，也就是说只动了 attention 的 Q、K、V、O 四个投影矩阵，FFN 完全跳过。
"""
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

# 把存下来的 LoRA 权重装回去
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)

# 把 LoRA 的"增量权重"单独存下来
# 存的是 A 和 B 的低秩矩阵，不是完整权重。rank=16 时，A 是 [16, 768]，B 是 [768, 16]——加起来比原矩阵 [768, 768] 小几十倍。
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'): # 只找贴了 LoRA 适配器的层（就是 attention 的 QKV、O 那四个投影）
            clean_name = name[7:] if name.startswith("module.") else name   #  DDP 包裹时会加前缀 module.，去掉它，让 key 干净
            # 取出 LoRA 的权重（A 矩阵和 B 矩阵，都是低秩的），搬回 CPU、转 fp16，存得小、推理时快
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            # 每找到一层就加到字典里
            state_dict.update(lora_state)
    torch.save(state_dict, path)


""" 为啥是W' = W + B·A？
    全参微调时，W_finetuned = W_pretrained + ΔW
    ΔW ≈ B @ A；B: [768, r]   A: [r, 768]
    W = W0 + B@A 的核心是"冻结预训练、只学一个小增量"，+ 保证起点不变、B@A 做低秩分解保证增量小但表达力够、两者同维保证合并后推理不冗余。"""
# 把 LoRA 和原模型合成一个完整权重
def merge_lora(model, lora_path, save_path):
    #  先把 LoRA 权重装到模型上
    load_lora(model, lora_path)
    #  拿到完整原始模型权重，但排除 LoRA 自己的 key
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        #  遍历所有线性层
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 从模型当前内存中的 weight 复制一份（这是个安全措施，防止修改原模型）
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                #  关键行——LoRA 的精髓就是 W' = W + B·A
                #  把低秩增量加到原 weight 上
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
