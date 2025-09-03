# evaluate.py (最终版, 智能加载，功能完整)
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import timm
import math

from vit_pruning import create_vit_with_pruning
from utils.data_loader import create_data_loaders
from utils.metrics import accuracy, calculate_efficiency_metrics, AverageMeter
from utils.lora import convert_to_lora_model

def setup_logging(rank):
    """一个简化的日志记录器"""
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - (rank:{rank}) - %(message)s')
    return logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model, loader, device):
    """验证集评估函数"""
    model.eval()
    top1_meter, top5_meter = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc="Validating", disable=(device.index != 0 if hasattr(device, 'index') and device.index is not None else False))
    for img, tgt in pbar:
        img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        out = model(img)
        acc1, acc5 = accuracy(out, tgt, topk=(1, 5))
        top1_meter.update(acc1.item(), img.size(0))
        top5_meter.update(acc5.item(), img.size(0))
    return top1_meter.avg, top5_meter.avg

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pruned ViT Models')
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint to evaluate')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=224, help='Image input size for model creation')
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(rank)

    logger.info(f"Loading checkpoint to evaluate: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    pruning_decisions = None
    
    # 智能加载逻辑
    if 'pruning_decisions' in ckpt and ckpt['pruning_decisions'] is not None:
        logger.info("Found 'pruning_decisions' in checkpoint. Building model directly.")
        pruning_decisions = ckpt['pruning_decisions']
        weights_to_load = ckpt['model_state_dict']
    elif 'args' in ckpt: # 兼容旧的搜索阶段checkpoint
        logger.warning("Could not find 'pruning_decisions' key. Rebuilding structure from search_checkpoint args...")
        train_args = ckpt['args']
        pruning_groups_str = getattr(train_args, 'pruning_groups', '')
        pruning_groups = [list(map(int, g.split(','))) for g in pruning_groups_str.split(';') if g]
        
        temp_model = create_vit_with_pruning(model_name=train_args.model, num_classes=1000, pruning_groups=pruning_groups)
        if hasattr(train_args, 'use_lora') and train_args.use_lora:
            temp_model = convert_to_lora_model(temp_model, rank=train_args.lora_rank, alpha=train_args.lora_alpha)
        
        temp_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        pruning_decisions = temp_model.get_pruning_decisions()
        weights_to_load = ckpt['model_state_dict']
    else:
        raise ValueError("Checkpoint is incomplete. Cannot determine model structure.")

    logger.info(f"Final Pruning Decisions: {pruning_decisions}")
    
    # 创建最终的静态学生模型
    student_model = timm.create_model("vit_base_patch16_224", num_classes=1000, depth=len(pruning_decisions), img_size=args.input_size)
    
    # 智能加载权重
    student_model.load_state_dict(weights_to_load, strict=False)
    student_model.to(device)
    logger.info("Weights loaded successfully.")

    _, val_loader = create_data_loaders("imagenet", args.data_dir, args.batch_size, args.num_workers, input_size=args.input_size)
    
    acc1, acc5 = evaluate(student_model, val_loader, device)
    efficiency_results = calculate_efficiency_metrics(student_model, input_shape=(1, 3, args.input_size, args.input_size), device=device)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluated Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print(f"Parameters (M): {efficiency_results['params'] / 1e6:.2f}")
    print(f"FLOPs (G): {efficiency_results['flops_g']:.2f}")
    print("="*80)

if __name__ == '__main__':
    main()