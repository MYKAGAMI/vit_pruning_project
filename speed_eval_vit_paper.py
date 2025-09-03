#!/usr/bin/env python
# speed_eval_vit_paper.py  (dtype-safe & EMA-list fix)

import argparse, time, torch, timm
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ---------------- accuracy util -----------------
def accuracy(out, tgt, topk=(1,)):
    _, pred = out.topk(max(topk), 1, True, True)
    corr    = pred.t().eq(tgt.view(1, -1))
    return [corr[:k].reshape(-1).float().sum() for k in topk]

# ---------------- load helper -------------------
def load_model(ckpt_tag, use_fp16=False):
    """支持  timm::<arch>  或 本地 ckpt(.pth)"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_tag.startswith("timm::"):
        arch = ckpt_tag.split("::")[1]
        model = timm.create_model(arch, pretrained=True, num_classes=1000).to(dev)
        if use_fp16: model.half()
        print(f"✓ timm model {arch} loaded")
        prune_dec = None

    else:  # 本地 checkpoint
        ckpt  = torch.load(ckpt_tag, map_location="cpu")
        prune_dec = ckpt.get("pruning_decisions", [])
        depth = len(prune_dec) if prune_dec else 12  # 默认 ViT-B/16 = 12
        model = timm.create_model("vit_base_patch16_224",
                                  depth=depth, num_classes=1000)
        # -------- 取权重（EMA > model_state_dict > root） --------
        if "ema" in ckpt and ckpt["ema"]:
            sd = ckpt["ema"].get("shadow_params", ckpt["ema"])
            #  shadow_params 可能是 list；转换成 dict
            if isinstance(sd, list):
                sd = {k: v for k, v in zip(model.state_dict().keys(), sd)}
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt

        # 去掉 module. 前缀
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        # 如果是裁剪模型，需要按 pruning_decisions 重映射层号
        if prune_dec:
            new_sd, depth = {}, len(prune_dec)
            for k in model.state_dict().keys():
                if "blocks." in k:
                    s_id = int(k.split('.')[1])
                    if s_id < depth:
                        t_id = prune_dec[s_id]
                        src_k = k.replace(f"blocks.{s_id}", f"blocks.{t_id}")
                        new_sd[k] = sd.get(src_k, sd.get(k))
                else:
                    new_sd[k] = sd.get(k)
            sd = new_sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"✓ ckpt loaded  layers={len(model.blocks)}  "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        model.to(dev)
        if use_fp16: model.half()

    model.eval()
    return model, dev

# ---------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help='timm::<arch> 或 本地 .pth')
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    model, dev = load_model(args.ckpt, use_fp16=args.fp16)

    # 1) FLOPs (thop)  —— dummy tensor 类型与模型保持一致
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 224, 224, device=dev)
        if args.fp16: dummy = dummy.half()
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs (224×224): {flops/1e9:.2f} G")
    except Exception as e:
        print(f"[thop] FLOPs 失败: {e}")

    # 2) 纯前向速度
    B = args.batch
    dummy = torch.randn(B, 3, 224, 224, device=dev)
    if args.fp16: dummy = dummy.half()
    warm = 20; rep = 100
    with torch.no_grad():
        for _ in range(warm):
            _ = model(dummy)
        torch.cuda.synchronize()
        st = time.time()
        for _ in range(rep):
            _ = model(dummy)
        torch.cuda.synchronize()
        et = time.time()
    avg_ms = (et - st) * 1000 / rep
    print(f"Speed (fp16={args.fp16})  batch {B}: {avg_ms:.2f} ms  "
          f"|  {1000*B/avg_ms:.2f} img/s")

if __name__ == "__main__":
    main()



