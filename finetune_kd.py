#!/usr/bin/env python
# finetune_kd.py —— KD + (可选) LoRA + AMP(fp16/bf16/fp32)

import os, argparse, logging, timm, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from vit_pruning import create_vit_with_pruning
from utils.data_loader   import create_data_loaders
from utils.scheduler     import get_cosine_schedule_with_warmup
from utils.metrics       import AverageMeter, accuracy
from utils.lora          import convert_to_lora_model, get_adapter_params


# --------------------------------------------------------------------------- #
#                             logging / DDP utils                             #
# --------------------------------------------------------------------------- #
def setup_logger(out_dir, rank):
    lg = logging.getLogger("kd_finetune")
    lg.setLevel(logging.INFO)
    if lg.handlers:
        return lg
    fmt = logging.Formatter("%(asctime)s - (rank:%(rank)s) - %(message)s",
                            defaults={"rank": rank})
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); lg.addHandler(sh)

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(out_dir, "finetune_kd.log"))
        fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def init_ddp():
    r = int(os.getenv("RANK", 0))
    w = int(os.getenv("WORLD_SIZE", 1))
    if w > 1 and not dist.is_initialized():
        dist.init_process_group("nccl"); torch.cuda.set_device(r)
    return r, w


# --------------------------------------------------------------------------- #
#                              evaluate & losses                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device, amp_enabled, amp_dtype, rank=0):
    model.eval()
    top1, top5 = AverageMeter(), AverageMeter()
    for img, tgt in tqdm(loader, desc="Val", disable=(rank != 0)):
        img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            out = model(img)
        a1, a5 = accuracy(out, tgt, (1, 5))
        top1.update(a1.item(), img.size(0)); top5.update(a5.item(), img.size(0))
    return top1.avg, top5.avg


def kd_loss(student_logits, teacher_logits, T=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean") * (T * T)


# --------------------------------------------------------------------------- #
def train_epoch(ep, stu, tea, loader,
                opt, sch, scaler, ema,
                args, lg, dev, rank,
                amp_enabled, amp_dtype):
    stu.train(); tea.eval()
    if dist.is_initialized():
        loader.sampler.set_epoch(ep)

    ce = nn.CrossEntropyLoss(
        label_smoothing=args.label_smooth_factor if args.label_smooth else 0.).to(dev)

    mloss, macc = AverageMeter(), AverageMeter()
    for it, (img, tgt) in enumerate(tqdm(loader, desc=f"E{ep}", disable=(rank != 0))):
        img, tgt = img.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            s_out = stu(img)
            with torch.no_grad():
                t_out = tea(img)
            loss = args.alpha * ce(s_out, tgt) + (1 - args.alpha) * kd_loss(
                s_out, t_out, args.temperature)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(stu.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if ema: ema.update()

        acc1, _ = accuracy(s_out, tgt, (1, 5))
        mloss.update(loss.item(), img.size(0)); macc.update(acc1.item(), img.size(0))
        if rank == 0 and it % args.log_interval == 0:
            lg.info(f"[E{ep} {it}/{len(loader)}] loss {mloss.avg:.4f} "
                    f"acc1 {macc.avg:.2f}")


# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser("KD finetune with unified aug / LoRA / AMP")

    # ckpt / io
    p.add_argument("--search_checkpoint", required=True)
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--teacher_arch", default="vit_base_patch16_224")
    p.add_argument("--resume")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)

    # train hparams
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=100)

    # KD
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)

    # tricks
    p.add_argument("--amp_dtype", choices=["fp16", "bf16", "float32"],
                   default="fp16")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--label_smooth", action="store_true")
    p.add_argument("--label_smooth_factor", type=float, default=0.1)
    p.add_argument("--save_strategy", choices=["best", "last", "both"],
                   default="best")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)

    # backward‑compat --fp16
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    if args.fp16:
        args.amp_dtype = "fp16"

    amp_enabled = args.amp_dtype != "float32"
    amp_dtype   = {"fp16": torch.float16,
                   "bf16": torch.bfloat16,
                   "float32": torch.float32}[args.amp_dtype]

    rank, world = init_ddp()
    lg   = setup_logger(args.output_dir, rank)
    dev  = torch.device(f"cuda:{rank}")

    # ---------------------------------------------------- #
    #                 build student network                #
    # ---------------------------------------------------- #
    if args.resume:
        ckpt   = torch.load(args.resume, map_location="cpu")
        prune  = ckpt["pruning_decisions"]

        stu = timm.create_model("vit_base_patch16_224",
                                depth=len(prune), num_classes=1000)
        stu.load_state_dict(ckpt["model_state_dict"])
        start_ep = ckpt["epoch"] + 1
        best     = ckpt["best_acc"]

    else:
        s_ckpt = torch.load(args.search_checkpoint, map_location="cpu")

        # --- Namespace → dict (兼容搜索 ckpt 的保存方式) ----------
        search_args = s_ckpt["args"]
        if not isinstance(search_args, dict):
            search_args = vars(search_args)
        # -----------------------------------------------------------

        groups = [list(map(int, g.split(",")))
                  for g in search_args["pruning_groups"].split(";") if g]

        tmp = create_vit_with_pruning(
                model_name=search_args["model"],
                pruning_groups=groups,
                num_classes=1000)
        tmp.load_state_dict(s_ckpt["model_state_dict"], strict=False)
        prune = tmp.get_pruning_decisions()

        stu = timm.create_model("vit_base_patch16_224",
                                depth=len(prune), num_classes=1000)

        # 把搜索模型权重复制到剪枝后的小模型
        sd_s, sd_t = stu.state_dict(), tmp.state_dict()
        for k in sd_s.keys():
            if "blocks." in k:
                sid = int(k.split(".")[1])
                if sid < len(prune):
                    src_k = k.replace(f"blocks.{sid}", f"blocks.{prune[sid]}")
                    if src_k in sd_t:
                        sd_s[k] = sd_t[src_k]
            elif k in sd_t:
                sd_s[k] = sd_t[k]
        stu.load_state_dict(sd_s)
        start_ep, best = 0, 0.0

    # ---------------- LoRA ---------------- #
    if args.use_lora:
        stu = convert_to_lora_model(stu,
                rank=args.lora_rank, alpha=args.lora_alpha)
        lg.info(f"LoRA enabled  rank={args.lora_rank}  alpha={args.lora_alpha}")

    stu.to(dev)

    # ---------------- teacher ------------- #
    tea = timm.create_model(args.teacher_arch, num_classes=1000, pretrained=False).to(dev)
    tea.load_state_dict(torch.load(args.teacher_checkpoint, map_location="cpu"), strict=False)
    tea.eval(); tea.requires_grad_(False)

    if world > 1:
        stu = DDP(stu, device_ids=[rank])

    # --------------- data / optim --------- #
    tr_loader, val_loader = create_data_loaders(
        "imagenet", args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=(world > 1),
        input_size=args.input_size)

    train_params = (get_adapter_params(stu) if args.use_lora else stu.parameters())
    opt = optim.AdamW(train_params, lr=args.lr, weight_decay=0.05)
    sch = get_cosine_schedule_with_warmup(
        opt,
        len(tr_loader) * args.warmup_epochs,
        len(tr_loader) * args.epochs)

    scaler = GradScaler(enabled=(args.amp_dtype == "fp16"))
    ema = ExponentialMovingAverage(stu.parameters(), decay=0.9995) if args.use_ema else None

    # ---------- resume optim/ema ---------- #
    if args.resume:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sch.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt.get("scaler_state", {}))
        if ema and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])

    # ----------------- loop --------------- #
    for ep in range(start_ep, args.epochs):
        train_epoch(ep, stu, tea, tr_loader,
                    opt, sch, scaler, ema,
                    args, lg, dev, rank,
                    amp_enabled, amp_dtype)

        if ema:
            ema.store(); ema.copy_to(stu.parameters())
        acc1, _ = validate(stu, val_loader, dev, amp_enabled, amp_dtype, rank)
        if ema:
            ema.restore()

        if rank == 0:
            lg.info(f"Epoch {ep} | acc1 {acc1:.2f}% (best {best:.2f}%)")

            save_now = (args.save_strategy in ["both", "last"]) or (acc1 > best)
            if save_now:
                mdl = stu.module if hasattr(stu, "module") else stu
                ckpt_to_save = {
                    "epoch": ep,
                    "best_acc": max(best, acc1),
                    "pruning_decisions": prune,
                    "model_state_dict": mdl.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sch.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "ema": (ema.state_dict() if ema else None),
                }
                torch.save(ckpt_to_save,
                           os.path.join(args.output_dir,
                                        "finetune_kd_best.pth" if acc1 > best
                                        else "finetune_kd_last.pth"))
            if acc1 > best:
                best = acc1

    if rank == 0:
        lg.info(f"Finish. Best Acc@1 {best:.2f}%")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()










