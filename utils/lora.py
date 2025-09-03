import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0, bias=True):
        super().__init__()
        self.in_features, self.out_features, self.rank, self.alpha = in_features, out_features, rank, alpha
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()
        self.is_lora = True
        self.weight.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return result + lora_update

def convert_to_lora_model(model, rank=8, alpha=16.0, target_modules=None):
    if target_modules is None: target_modules = ["qkv", "proj"]
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not hasattr(module, 'is_lora'):
            if any(target_name in name for target_name in target_modules):
                parent_module = model.get_submodule(".".join(name.split('.')[:-1]))
                lora_layer = LoRALinear(module.in_features, module.out_features, rank, alpha, bias=module.bias is not None)
                lora_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None: lora_layer.bias.data.copy_(module.bias.data)
                lora_layer.to(module.weight.device)
                setattr(parent_module, name.split('.')[-1], lora_layer)
    return model

def get_adapter_params(model):
    return [p for name, p in model.named_parameters() if 'lora_' in name and p.requires_grad]