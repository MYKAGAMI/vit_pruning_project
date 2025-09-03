import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val, self.sum, self.count = val, self.sum + val * n, self.count + n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk, batch_size = max(topk), target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]

def calculate_flops(model, input_shape, device):
    try:
        from thop import profile
        flops, _ = profile(model.to(device), inputs=(torch.randn(input_shape).to(device),), verbose=False)
        return int(flops)
    except Exception as e:
        print(f"  [Warning] FLOPs calculation with thop failed: {e}. Returning 0.")
        return 0

def calculate_efficiency_metrics(model, input_shape=(1, 3, 224, 224), device=None):
    if device is None: device = next(model.parameters()).device
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    flops = calculate_flops(model, input_shape, device=device)
    return {'params': params, 'flops_g': flops / 1e9}