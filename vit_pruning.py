# vit_pruning.py  (仅改了 gate requires_grad=True；其余保持一致)
import torch, torch.nn as nn, torch.nn.functional as F, itertools
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer

def generate_binary_tensor(N: int, M: int) -> torch.Tensor:
    if N > M:
        raise ValueError(f"N={N} > M={M}")
    combos = list(itertools.combinations(range(M), N))
    out = torch.zeros((len(combos), M), dtype=torch.float32)
    for i, idx in enumerate(combos):
        out[i, torch.tensor(idx)] = 1
    return out

class ViTWithPruning(VisionTransformer):
    def __init__(self, pruning_groups: Optional[List[List[int]]] = None, **kw):
        super().__init__(**kw)
        self.depth = kw.get("depth", 12)
        self.pruning_decisions = None

        self.groups = pruning_groups or [[1, 2]] * (self.depth // 2)
        self._setup_pruning_components()
        self.tau, self.scaling = 2.0, 1.0

    # ---------------------------------------------------------------- #
    def _setup_pruning_components(self):
        assert sum(M for _, M in self.groups) == self.depth
        opts, gates = [], []
        for N, M in self.groups:
            opt = generate_binary_tensor(N, M)
            opts.append(opt)
            gate = nn.Parameter(torch.randn(1, opt.shape[0]),  # ← trainable!
                                requires_grad=True)
            nn.init.constant_(gate, 0.02)
            gates.append(gate)
        self.options = nn.ParameterList(
            [nn.Parameter(o, requires_grad=False) for o in opts]
        )
        self.gumbel_gates = nn.ParameterList(gates)

    # ---------------------------------------------------------------- #
    def forward_features(self, x):
        x = self.patch_embed(x); x = self._pos_embed(x); x = self.pos_drop(x)

        if self.training:
            layer_id = 0
            B = x.size(0)
            for g_idx, gate in enumerate(self.gumbel_gates):
                opt = self.options[g_idx].to(gate.device)
                logits = gate.repeat(B, 1) * self.scaling
                samp = F.gumbel_softmax(logits, tau=self.tau,
                                        hard=False, dim=1)
                mask = samp @ opt        # B × M
                _, M = self.groups[g_idx]
                for j in range(M):
                    if layer_id >= len(self.blocks): break
                    out = self.blocks[layer_id](x)
                    m = mask[:, j].view(B, 1, 1)
                    x = out * m + x * (1 - m)
                    layer_id += 1
        else:
            blocks = (self.blocks if self.pruning_decisions is None
                      else [self.blocks[i] for i in self.pruning_decisions])
            for blk in blocks: x = blk(x)

        return self.norm(x)

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def get_pruning_decisions(self):
        if not self.gumbel_gates: return list(range(len(self.blocks)))
        dec, offset = [], 0
        for (N, M), gate, opt in zip(self.groups, self.gumbel_gates, self.options):
            idx = gate.argmax(dim=1).item()
            chosen = opt[idx].nonzero().squeeze(-1)
            dec += (chosen + offset).tolist()
            offset += M
        return sorted(dec)

    @torch.no_grad()
    def get_gate_probabilities(self):
        return [F.softmax(g * self.scaling, 1).squeeze().cpu()
                for g in self.gumbel_gates]

# --------------------------------------------------------------------- #
def create_vit_with_pruning(model_name="vit_base_patch16_224", **kw):
    cfg = {"vit_base_patch16_224":
           dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)}
    if model_name not in cfg: raise ValueError(model_name)
    pruning_groups = kw.pop("pruning_groups", None)
    return ViTWithPruning(pruning_groups, **cfg[model_name], **kw)
