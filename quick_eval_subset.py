#!/usr/bin/env python
# quick_eval_subset.py —— 轻量评测 (剪枝 / LoRA / EMA / AMP)

import argparse, random, torch, timm, torchvision as tv
from torch.utils.data import DataLoader, Subset
from timm.data import resolve_data_config, create_transform
from torch.amp import autocast           # ≥ PyTorch‑2.1
from tqdm import tqdm


# --------------------------- 小工具 --------------------------- #
def accuracy(out, tgt, topk=(1,)):
    _, pred = out.topk(max(topk), 1, True, True)
    corr = pred.t().eq(tgt.view(1, -1))
    return [corr[:k].reshape(-1).float().sum() for k in topk]


def _inject_lora(model, sd):
    """若权重里有 lora_A/lora_B，动态插 LoRA 层"""
    if not any("lora_" in k for k in sd):
        return model
    from utils.lora import convert_to_lora_model
    rank = next(v.size(0) for k, v in sd.items() if "lora_A" in k)
    alpha = rank * 2
    model = convert_to_lora_model(model, rank=rank, alpha=alpha)
    print(f"✓ LoRA injected (rank={rank}, alpha≈{alpha})")
    return model


def _load_ema(model, ema_blob) -> bool:
    """兼容新旧 torch‑ema: 返回 True=加载成功"""
    if not isinstance(ema_blob, dict):
        return False

    # 1) >=0.5: {'decay':..,'num_updates':..,'shadow_params':[tensor,…]}
    # 2) state_dict: {'state_dict':{'shadow_params':…}}
    sp = (ema_blob.get("shadow_params") or
          ema_blob.get("params") or
          (ema_blob.get("state_dict", {}).get("shadow_params")))

    if sp is None:
        return False

    if isinstance(sp, dict) and all(isinstance(k, str) for k in sp):
        # full name → 直接 load_state_dict
        clean = {k.lstrip("module."): v for k, v in sp.items()}
        model.load_state_dict(clean, strict=False)
        return True

    if isinstance(sp, (list, tuple)) or (
        isinstance(sp, dict) and all(isinstance(k, int) for k in sp)
    ):
        # list / int‑key dict → 按参数顺序 copy
        seq = sp.values() if isinstance(sp, dict) else sp
        for p, v in zip(model.parameters(), seq):
            p.data.copy_(v.to(p.dtype))
        return True

    return False


def load_model(ckpt_path: str, amp_dtype: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    prune = ckpt.get("pruning_decisions", [])
    depth = len(prune) if prune else 12
    model = timm.create_model("vit_base_patch16_224",
                              num_classes=1000, depth=depth)

    raw_sd = ckpt.get("model_state_dict", ckpt)
    miss, _ = model.load_state_dict(raw_sd, strict=False)

    # ---- 剪枝权重 remap (老 search_ckpt) ----
    if miss and prune:
        new_sd = model.state_dict()
        for k in new_sd:
            if "blocks." in k:
                sid = int(k.split('.')[1])
                if sid < len(prune):
                    src = k.replace(f"blocks.{sid}", f"blocks.{prune[sid]}")
                    if src in raw_sd:
                        new_sd[k] = raw_sd[src]
            elif k in raw_sd:
                new_sd[k] = raw_sd[k]
        model.load_state_dict(new_sd, strict=True)

    # ---- LoRA ----
    model = _inject_lora(model, raw_sd)

    # ---- EMA ----
    ok = _load_ema(model, ckpt.get("ema"))
    print("✓ EMA loaded" if ok else "✗ EMA not found – using student weights")

    # ---- AMP dtype / device ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    if amp_dtype == "fp16":
        model.half()
    elif amp_dtype == "bf16":
        model.to(dtype=torch.bfloat16)
    return model, dev


# --------------------------- 主函数 --------------------------- #
def main():
    ap = argparse.ArgumentParser("ViT quick eval")
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--valdir", required=True)
    ap.add_argument("--subset", type=int, default=0,
                    help="随机抽样数目 (0=全量)")
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16", "float32"],
                    default="float32")
    ap.add_argument("--train_aug", action="store_true",
                    help="验证也用训练增强 (RandomResizedCrop + Flip)")
    ap.add_argument("--views", type=int, default=1,
                    help="多视图；>1 时额外水平翻转")
    args = ap.parse_args(); assert args.views >= 1

    model, dev = load_model(args.ckpt, args.amp_dtype)

    # ------------ 数据集 ------------ #
    cfg = resolve_data_config({}, model=model)
    trans = create_transform(
        input_size=cfg["input_size"],
        is_training=args.train_aug,
        auto_augment=cfg.get("auto_augment") if args.train_aug else None,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"], std=cfg["std"],
    )

    base = tv.datasets.ImageFolder(args.valdir, trans)
    if 0 < args.subset < len(base):
        idx = random.sample(range(len(base)), args.subset)
        ds = Subset(base, idx); print(f"Subset {len(ds):,}/{len(base):,}")
    else:
        ds = base; print(f"Full set {len(ds):,}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # ------------ 推理 ------------ #
    top1 = top5 = seen = 0
    use_amp  = args.amp_dtype in ["fp16", "bf16"]
    amp_type = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp_dtype)

    pbar = tqdm(loader, total=len(loader), unit="batch")
    with torch.no_grad():
        for img, tgt in pbar:
            img, tgt = img.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            if args.amp_dtype == "fp16":
                img = img.half()
            elif args.amp_dtype == "bf16":
                img = img.to(dtype=torch.bfloat16)

            logits = 0
            for v in range(args.views):
                x = img if v == 0 else torch.flip(img, dims=[3])
                with autocast("cuda", enabled=use_amp, dtype=amp_type):
                    logits += model(x)
            out = logits / args.views

            c1, c5 = accuracy(out, tgt, (1, 5))
            top1 += c1.item(); top5 += c5.item(); seen += tgt.size(0)
            pbar.set_postfix(top1=f"{100*top1/seen:5.2f}%",
                             top5=f"{100*top5/seen:5.2f}%")

    print(f"\nFinal  Top‑1 {100*top1/seen:.2f}%   Top‑5 {100*top5/seen:.2f}%")


if __name__ == "__main__":
    main()







