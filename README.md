# ViT Pruning Project

Research code for **Vision Transformer (ViT)** pruning with **Knowledge Distillation (KD)**.  
Includes KD variants (LoRA-KD / EMA), simple pruning/search utilities, and accuracy/speed evaluation.

> **Status:** research code, evolving  
> **Primary dataset:** ImageNet-1K  
> **Repository:** https://github.com/MYKAGAMI/vit_pruning_project

---

## Highlights

- **Structured pruning** for ViT (rule-based and simple search recipes).
- **Knowledge Distillation**: standard KD, **LoRA-KD**, **EMA** variants.
- **Evaluation**: ImageNet accuracy (full/subset) and simple latency/throughput checks.
- Lightweight, script-first workflow built on PyTorch + `timm`.

---

## Repository Layout (key files)


> **Note:** Large artifacts (e.g., `*.pth`, logs, `runs/`, `wandb/`, `outputs/`) should be ignored via `.gitignore`.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (match your CUDA version)
- Packages: `timm`, `torchvision`, `numpy`, `tqdm`, `pyyaml`

Example install (adjust CUDA wheel URL as needed):
```bash
# PyTorch (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Other deps
pip install timm numpy tqdm pyyaml

Optional virtual environment:

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

Data Preparation

Assumes ImageNet-1K with the usual layout:

/path/to/imagenet
├── train/
└── val/


Pass the path with --data /path/to/imagenet.

Prepare Teacher Weights

Download or place teacher checkpoints (e.g., BEiTv2, DeiT-III) locally.

python download_teacher.py --model beitv2 --out ./teacher_model_beitv2.pth
# See available options:
python download_teacher.py -h

Quick Start
1) Baseline fine-tuning (no KD)
python finetune_kd_noKD/finetune_kd_noKD.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --epochs 50 --batch-size 128 --lr 5e-4 \
  --output ./outputs/baseline

2) Standard Knowledge Distillation
python finetune_kd.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --teacher ./teacher_model_beitv2.pth \
  --epochs 50 --batch-size 128 \
  --output ./outputs/kd

3) Variants: LoRA-KD / EMA
# LoRA-KD
python finetune_kd_lora_kd/finetune_kd_lora_kd.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --teacher ./teacher_model_beitv2.pth \
  --output ./outputs/lora_kd

# EMA
python finetune_kd_ema/finetune_kd_ema.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --teacher ./teacher_model_beitv2.pth \
  --output ./outputs/kd_ema


Use -h/--help on each script for full arguments (e.g., --seed, optimizer/scheduler, KD temperature --T, KD weight --alpha, AMP, etc.).

Pruning
A) Rule-based pruning → pruned checkpoint
python make_manual_prune_ckpt.py \
  --in_ckpt ./outputs/kd/checkpoint_student.pth \
  --rule "keep=5/6" \
  --out_ckpt ./outputs/kd/checkpoint_student_pruned.pth

B) Search a pruning policy (example)
python search_final_run/run_search.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --target_flops_ratio 0.70 \
  --output ./outputs/search_70

C) Direct pruning helper
python vit_pruning.py \
  --in_ckpt ./outputs/kd/checkpoint_student.pth \
  --strategy layer --ratio 0.75 \
  --out_ckpt ./outputs/kd/checkpoint_student_pruned.pth

Evaluation
Accuracy (full or subset)
python evaluate.py \
  --data /path/to/imagenet \
  --arch vit_base_patch16_224 \
  --ckpt ./outputs/kd/checkpoint_student_pruned.pth \
  --batch-size 256

Quick subset evaluation
python quick_eval_subset.py \
  --data /path/to/imagenet/val_subset \
  --ckpt ./outputs/kd/checkpoint_student_pruned.pth

Latency / throughput (simple)
python speed_eval_vit_paper.py \
  --arch vit_base_patch16_224 \
  --ckpt ./outputs/kd/checkpoint_student_pruned.pth \
  --batch-size 1 --repeat 200
