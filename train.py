#!/usr/bin/env python
# train.py —— ViT learnable pruning *search phase*
# ✅ 支持 LoRA / bf16‑AMP / fp16‑AMP / fp32
# ✅ 两套 AdamW（主干 & gate）+ Cosine 调度
# ✅ 每个 epoch 打印剪枝 Decisions / Ratio / Gate Probabilities
# ✅ 同时保存 search_best.pth  &  search_last.pth

import os, argparse, logging, torch, torch.nn as nn, torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import timm

from vit_pruning import create_vit_with_pruning
from utils.data_loader import create_data_loaders
from utils.lora       import convert_to_lora_model, get_adapter_params
from utils.metrics    import AverageMeter, accuracy
from utils.scheduler  import get_cosine_schedule_with_warmup


# --------------------------------------------------------------------------- #
#                           logging / distributed utils                       #
# --------------------------------------------------------------------------- #
def setup_logger(out_dir: str, rank: int) -> logging.Logger:
    lg = logging.getLogger("train_search"); lg.setLevel(logging.INFO)
    if not lg.handlers:                                  # 防止重复添加
        fmt = logging.Formatter("%(asctime)s - (rank:%(rank)s) - %(message)s",
                                defaults={"rank": rank})
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(out_dir, "train_search.log"))
            fh.setFormatter(fmt); lg.addHandler(fh)
            sh = logging.StreamHandler();  sh.setFormatter(fmt); lg.addHandler(sh)
        else:
            lg.addHandler(logging.NullHandler())
    return lg


def init_ddp():
    rank  = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl"); torch.cuda.set_device(rank)
    return rank, world


# --------------------------------------------------------------------------- #
#                               eval & helpers                                #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device, amp_enabled, amp_dtype, rank=0):
    model.eval()
    t1, t5 = AverageMeter(), AverageMeter()
    for img, tgt in tqdm(loader, desc="Val", disable=(rank != 0)):
        img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            out = model(img)
        a1, a5 = accuracy(out, tgt, (1, 5))
        t1.update(a1.item(), img.size(0)); t5.update(a5.item(), img.size(0))
    return t1.avg, t5.avg


def log_pruning_status(model, logger):
    """打印 Decisions / Ratio / Gate Probabilities（每个 epoch 调一次）"""
    if not hasattr(model, "get_pruning_decisions"):
        return
    dec = model.get_pruning_decisions()
    ratio = 1 - len(dec) / getattr(model, "depth", len(model.blocks))
    logger.info("--- Pruning Status ---")
    logger.info(f"Decisions: {dec}")
    logger.info(f"Ratio    : {ratio:.2%}")
    if hasattr(model, "get_gate_probabilities"):
        logger.info("Gate Probabilities:")
        for i, p in enumerate(model.get_gate_probabilities()):
            logger.info(f"  Group {i}: {[f'{x:.3f}' for x in p.tolist()]}")
    logger.info("----------------------")


# --------------------------------------------------------------------------- #
#                                 train step                                  #
# --------------------------------------------------------------------------- #
def train_epoch(ep, model, loader,
                opt_main, opt_gate, sched_main,
                scaler, args, logger, rank,
                amp_enabled, amp_dtype):

    model.train()
    if dist.is_initialized():
        loader.sampler.set_epoch(ep)

    # temperature & scaling 退火
    mdl_no_ddp = model.module if hasattr(model, "module") else model
    progress   = ep / args.epochs
    if hasattr(mdl_no_ddp, "tau"):
        mdl_no_ddp.tau     = args.tau_max * (args.tau_min / args.tau_max) ** progress
        mdl_no_ddp.scaling = args.scaling_min + (args.scaling_max - args.scaling_min) * progress

    ce = nn.CrossEntropyLoss().to(rank)
    mloss, macc = AverageMeter(), AverageMeter()

    for i, (img, tgt) in enumerate(tqdm(loader, desc=f"E{ep}", disable=(rank != 0))):
        img, tgt = img.cuda(rank, non_blocking=True), tgt.cuda(rank, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            out  = model(img)
            loss = ce(out, tgt)

        opt_main.zero_grad();  opt_gate.zero_grad()

        if scaler is not None:        # fp16
            scaler.scale(loss).backward()
            scaler.unscale_(opt_main); scaler.unscale_(opt_gate)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt_main); scaler.step(opt_gate); scaler.update()
        else:                         # bf16 / fp32
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt_main.step(); opt_gate.step()

        sched_main.step()

        a1, _ = accuracy(out, tgt, topk=(1, 5))
        mloss.update(loss.item(), img.size(0)); macc.update(a1.item(), img.size(0))

        if rank == 0 and i % args.log_interval == 0:
            lrm = sched_main.get_last_lr()[0]; lrg = opt_gate.param_groups[0]["lr"]
            logger.info(f"E{ep} {i}/{len(loader)} "
                        f"loss {mloss.avg:.4f}  acc1 {macc.avg:.2f}% "
                        f"LR_main {lrm:.2e}  LR_gate {lrg:.2e}")


# --------------------------------------------------------------------------- #
#                                   argparser                                 #
# --------------------------------------------------------------------------- #
def get_args():
    ap = argparse.ArgumentParser("ViT Pruning Search (LoRA / AMP)")
    ap.add_argument("--model", default="vit_base_patch16_224")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pruning_groups", default="3,4;3,4;3,4")
    ap.add_argument("--num_classes", type=int, default=1000)

    # pruning hyper‑params
    ap.add_argument("--tau_max", type=float, default=2.0)
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--scaling_min", type=float, default=1.0)
    ap.add_argument("--scaling_max", type=float, default=100.)

    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=100)

    # AMP
    ap.add_argument("--amp_dtype", default="bf16",
                    choices=["bf16", "fp16", "float32"],
                    help="bf16 (推荐 A/H100) / fp16 / float32")
    # timm pretrained
    ap.add_argument("--use_timm_pretrained", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    rank, world = init_ddp()
    lg   = setup_logger(args.output_dir, rank)
    dev  = torch.device(f"cuda:{rank}")

    # ---------- build student (search phase model) ----------
    pg = [list(map(int, g.split(","))) for g in args.pruning_groups.split(";") if g]
    model = create_vit_with_pruning(model_name=args.model,
                                    num_classes=args.num_classes,
                                    pruning_groups=pg)

    # timm pretrained
    if args.use_timm_pretrained and rank == 0:
        lg.info(f"Load timm pretrained for {args.model}")
    if args.use_timm_pretrained:
        tm = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
        model.load_state_dict(tm.state_dict(), strict=False); del tm

    # LoRA
    if args.use_lora:
        model = convert_to_lora_model(model,
                                      rank=args.lora_rank,
                                      alpha=args.lora_alpha)
        if rank == 0:
            lg.info("LoRA injected.")

    model.to(dev)

    # ---------- parameter groups ----------
    gate_p   = [p for n, p in model.named_parameters() if "gumbel_gates" in n]
    main_p   = (get_adapter_params(model) if args.use_lora
                else [p for n, p in model.named_parameters()
                      if "gumbel_gates" not in n])

    opt_main = optim.AdamW(main_p, lr=args.lr,  weight_decay=args.weight_decay)
    opt_gate = optim.AdamW(gate_p, lr=args.gate_lr, weight_decay=0.)

    # ---------- dataloader ----------
    tr_loader, val_loader = create_data_loaders(
        "imagenet", args.data_dir, args.batch_size, args.num_workers,
        distributed=(world > 1))

    sched_main = get_cosine_schedule_with_warmup(
        opt_main,
        int(len(tr_loader) * args.epochs * args.warmup_ratio),
        len(tr_loader) * args.epochs)

    if world > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    amp_enabled = args.amp_dtype != "float32"
    amp_dtype   = {"bf16": torch.bfloat16,
                   "fp16": torch.float16,
                   "float32": torch.float32}[args.amp_dtype]
    scaler = (GradScaler(enabled=(args.amp_dtype == "fp16"))
              if args.amp_dtype == "fp16" else None)

    best = 0.0
    for ep in range(args.epochs):
        train_epoch(ep, model, tr_loader,
                    opt_main, opt_gate, sched_main,
                    scaler, args, lg, rank,
                    amp_enabled, amp_dtype)

        acc1, _ = validate(model, val_loader, dev,
                           amp_enabled, amp_dtype, rank)

        if rank == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            lg.info(f"Epoch {ep}  acc1 {acc1:.2f}%  (best {best:.2f}%)")
            log_pruning_status(model_to_save, lg)

            # checkpoint 统一内容
            ckpt = {
                "epoch": ep,
                "best_acc": best,
                "val_acc": acc1,
                "model_state_dict": model_to_save.state_dict(),
                "pruning_decisions": model_to_save.get_pruning_decisions(),
                "args": vars(args),
            }

            # 1) 始终覆盖保存“最后一轮”
            torch.save(ckpt,
                       os.path.join(args.output_dir, "search_last.pth"))

            # 2) 若刷新最佳精度，额外存 best
            if acc1 > best:
                best = acc1
                ckpt["best_acc"] = best
                torch.save(ckpt,
                           os.path.join(args.output_dir, "search_best.pth"))
                lg.info("✓ saved new best")

    if rank == 0:
        lg.info(f"Finish. Best Acc@1 {best:.2f}%")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()




