#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动生成“search-phase”风格的 ckpt，方便做
  · 均匀剪 / 随机剪
  · 直接丢给 finetune_kd.py 继续微调
"""

import argparse, random, torch, timm
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vit_base_patch16_224")
    ap.add_argument("--pretrained", action="store_true",
                    help="直接使用 timm 的 ImageNet 权重")
    ap.add_argument("--load_ckpt", type=str, default=None,
                    help="自定义全参 ckpt（优先级高于 --pretrained）")
    ap.add_argument("--group_size", type=int, default=6,
                    help="M：每组多少层")
    ap.add_argument("--keep_per_group", type=int, default=5,
                    help="N：每组保留多少层")
    ap.add_argument("--mode", choices=["uniform", "random"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="输出 ckpt 文件")
    return ap.parse_args()

def build_full_model(args):
    if args.load_ckpt:
        print(f"[INFO] load full model from {args.load_ckpt}")
        full = timm.create_model(args.arch, num_classes=1000)
        sd = torch.load(args.load_ckpt, map_location="cpu")
        # 兼容 ‘model_state_dict’ 或直接权重
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        full.load_state_dict(sd, strict=False)
    else:
        full = timm.create_model(
            args.arch, pretrained=args.pretrained, num_classes=1000
        )
    return full

def make_pruning_decisions(depth, M, N, mode, seed=42):
    assert depth % M == 0, f"{depth=} 不能整除 {M=}"
    groups = depth // M
    decisions = []
    rng = random.Random(seed)
    for g in range(groups):
        start = g * M
        if mode == "uniform":
            # 均匀：把区间均分成 M 段，删除第 (M-N) 个索引
            drop_idx = start + (g % M)  # 也可换别的策略
        else:  # random
            drop_idx = start + rng.choice(range(M))
        keep = [i for i in range(start, start + M) if i != drop_idx]
        # 如果要一次删多层，可扩展这里
        assert len(keep) == N
        decisions.extend(keep)
    decisions.sort()
    return decisions  # 长度 = groups*N

def main():
    args = parse_args()
    full = build_full_model(args)
    depth = len(full.blocks)
    print(f"[INFO] full-model depth = {depth}")

    M, N = args.group_size, args.keep_per_group
    assert N < M, "N must be < M"
    pruned_layers = make_pruning_decisions(depth, M, N, args.mode, args.seed)
    print(f"[INFO] pruning_decisions ({len(pruned_layers)} layers kept): {pruned_layers}")

    # 构造新模型骨架
    student = timm.create_model(args.arch, num_classes=1000, depth=len(pruned_layers))
    sd_s, sd_f = student.state_dict(), full.state_dict()

    # 把保留下来的层权重 copy 过去
    for k in sd_s:
        if "blocks." in k:
            s_idx = int(k.split('.')[1])
            f_idx = pruned_layers[s_idx]
            src_key = k.replace(f"blocks.{s_idx}", f"blocks.{f_idx}")
            sd_s[k] = sd_f[src_key]
        elif k in sd_f:
            sd_s[k] = sd_f[k]
    student.load_state_dict(sd_s)

    # 组成 pseudo “search ckpt”
    ckpt = {
        "epoch": 0,
        "best_acc": 0.0,
        "model_state_dict": student.state_dict(),
        "pruning_decisions": pruned_layers,
        "args": {"model": args.arch, "pruning_groups": ""},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[OK] Saved to {args.out}")

if __name__ == "__main__":
    main()
