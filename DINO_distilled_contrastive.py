# Full patched version will be inserted here.
# Please paste your original script content below so patches can be applied accurately.
# (Due to length, you should paste the full original code here and I will apply the patches.)
#!/usr/bin/env python3
"""
kd_distill_contrastive.py
Memory-safe Knowledge Distillation script (single-file) using **Contrastive Loss**.
MODIFICATIONS:
- Implements Multi-View Augmentation (v1, v2) for positive pairs.
- Implements Momentum Encoder for the Teacher (BYOL/DINO style).
- Fixes TIMM model creation to be compatible with 96x96 inputs when loading pretrained weights.

Key features:
 - Contrastive Knowledge Distillation (CKD) based on student-teacher projection similarity.
 - Different seeds for teacher vs student initialization (prevents KD=0).
 - proj_dim default 1024 (avoids giant projection matrices).
 - Mixed precision via torch.amp.
 - Gradient accumulation supported.
 - Optional timm backbones for teacher/student (teacher_pretrained optional).
 - Simple k-NN eval using frozen student encoder and ImageFolder eval_public splits.

Usage example (unlabeled CKD with momentum teacher and 96x96 ViT):
  python kd_distill_contrastive.py --data_dir dataset_cache/unzipped \
      --proj_dim 1024 --contrastive --contrastive_weight 1.0 --epochs 200 --batch_size 128 --amp \
      --use_timm_teacher --teacher_model vit_base_patch16_224 --teacher_pretrained
"""

import os
import time
import math
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import plotly.graph_objects as go
import json


# optional
try:
    import timm
except Exception:
    timm = None

# -------------
# Utilities
# -------------
def exists(x): return x is not None

def update_teacher_momentum(student_model, teacher_model, m):
    """
    Performs Exponential Moving Average (EMA) update on teacher weights.
    teacher_weights = m * teacher_weights + (1 - m) * student_weights
    
    Uses name-based matching to handle different architectures (e.g., vit_base vs vit_small).
    Only updates parameters that exist in both models and have matching shapes.
    """
    student_dict = dict(student_model.named_parameters())
    teacher_dict = dict(teacher_model.named_parameters())
    
    updated_count = 0
    skipped_count = 0
    for name, param_t in teacher_dict.items():
        if name in student_dict:
            param_s = student_dict[name]
            if param_s.shape == param_t.shape:
                param_t.data.mul_(m).add_(param_s.data, alpha=1.0 - m)
                updated_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    # Only print warning if many parameters were skipped (indicates architecture mismatch)
    if skipped_count > 0 and updated_count == 0:
        print("  ⚠️ Warning: No parameters matched between teacher and student (different architectures). EMA update skipped.")

# -------------
# Small conv backbone (memory-friendly fallback)
# -------------
class TinyConvBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------
# Head / projector
# -------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1024, nlayers=2, use_norm=True):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(nlayers-1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.GELU())
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

# -------------
# Model wrapper (backbone + head)
# -------------
class ModelWrapper(nn.Module):
    def __init__(self, use_timm=False, timm_name=None, embed_dim=256, proj_dim=1024, img_size=96): # Added img_size=96
        super().__init__()
        self.use_timm = use_timm and (timm is not None)
        if self.use_timm:
            assert timm_name is not None, "timm_name required when use_timm=True"
            # create timm model returning features; num_classes=0 gives features
            # 💡 FIX: Added img_size to configure ViT backbone for 96x96 input
            backbone = timm.create_model(timm_name, pretrained=False, num_classes=0, img_size=img_size) 
            self.backbone = backbone
            out_dim = getattr(backbone, 'num_features', embed_dim)
        else:
            self.backbone = TinyConvBackbone(out_dim=embed_dim)
            out_dim = embed_dim
        self.embed_dim = out_dim
        self.proj_dim = proj_dim
        self.head = MLPHead(in_dim=out_dim, out_dim=proj_dim)
        print(f"[MODELWRAP] use_timm={self.use_timm} backbone_out={out_dim} proj_dim={proj_dim} img_size={img_size}")

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim == 4:
            # some timm models return spatial maps; global average
            feat = feat.mean(dim=(-2, -1))
        proj = self.head(feat)
        return feat, proj

# -------------
# Dataset builder (MODIFIED for Dual Augmentation)
# -------------

# 💡 NEW CLASS: Applies two independent transforms for v1 and v2
class DualAugmentationTransform:
    """Applies two independent, random augmentations to the same input image."""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        v1 = self.transform(x)
        v2 = self.transform(x)
        return v1, v2

def build_dataset(data_dir=None, hf_dataset=None):
    # Using 96x96 for consistent input size with ViT-96 models
    # Stronger augmentations for better contrastive learning
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),             
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if data_dir is None:
        raise RuntimeError("Please provide --data_dir pointing to ImageFolder of images.")

    # Apply the DualAugmentationTransform to get two views per image
    dual_transform = DualAugmentationTransform(base_transform)
    ds = datasets.ImageFolder(data_dir, transform=dual_transform)
    return ds

# -------------
# Loss helpers (MODIFIED for Contrastive Loss)
# -------------

def kl_div_logits(student_logits, teacher_logits, T):
    # retained for supervised CE (if needed) but not main KD loss
    if T <= 0:
        return torch.tensor(0.0, device=student_logits.device)
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1).detach()
    return F.kl_div(s_log, t_prob, reduction='batchmean') * (T * T)

def contrastive_loss(s_proj, t_proj, T=0.07, symmetric=True):
    # Student and teacher projections (s_proj, t_proj) are positive pairs.
    # Dimensions: (B, D). B is batch size, D is projection dimension.
    # We treat all other pairs in the batch as negative. This is InfoNCE loss.

    # 1. Normalize the projections
    z_s = F.normalize(s_proj, dim=1)
    z_t = F.normalize(t_proj, dim=1).detach() # Detach teacher to stop gradient flow

    # 2. Compute similarity matrix
    # sim_matrix[i, j] = dot product (cosine similarity) between z_s[i] and z_t[j]
    similarity_matrix = torch.matmul(z_s, z_t.T) / T

    # 3. Targets (Positive pairs are the diagonal)
    targets = torch.arange(z_s.size(0), device=z_s.device)
    
    # Loss is computed for z_s against z_t (as positives)
    loss_s2t = F.cross_entropy(similarity_matrix, targets)
    
    if symmetric:
        # Symmetric loss: z_t against z_s
        loss_t2s = F.cross_entropy(similarity_matrix.T, targets)
        loss = (loss_s2t + loss_t2s) / 2.0
    else:
        loss = loss_s2t

    return loss

# -------------
# k-NN eval (simple cosine k-NN)
# -------------
def knn_eval(student, eval_public_dir, device, args):
    print("[k-NN] Running k-NN using eval_public at", eval_public_dir)
    train_dir = os.path.join(eval_public_dir, 'train')
    test_dir = os.path.join(eval_public_dir, 'test')
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print("[k-NN] eval_public must contain train/ and test/ ImageFolder subfolders. Skipping k-NN.")
        return

    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.CenterCrop(96),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    student.eval()
    bank_feats = []
    bank_labels = []
    with torch.no_grad():
        # NOTE: k-NN eval uses the default ImageFolder transform, which yields one image (imgs)
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feat, proj = student(imgs)
            f = feat
            f = F.normalize(f, dim=-1)
            bank_feats.append(f.cpu())
            bank_labels.append(labels)
        bank_feats = torch.cat(bank_feats, dim=0).numpy()
        bank_labels = torch.cat(bank_labels, dim=0).numpy()

        test_feats = []
        test_labels = []
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feat, proj = student(imgs)
            f = F.normalize(feat, dim=-1)
            test_feats.append(f.cpu())
            test_labels.append(labels)
        test_feats = torch.cat(test_feats, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()

    # cosine similarities via dot product (features are normalized)
    print("[k-NN] Computing predictions (k=%d) ..." % args.knn_k)
    sims = test_feats @ bank_feats.T  # (Ntest, Nbank)
    topk_idx = np.argpartition(-sims, args.knn_k, axis=1)[:, :args.knn_k]  # faster than topk for numpy
    preds = []
    for i in range(test_feats.shape[0]):
        idxs = topk_idx[i]
        votes = bank_labels[idxs]
        pred = Counter(votes).most_common(1)[0][0]
        preds.append(pred)
    preds = np.array(preds)
    acc = (preds == test_labels).mean() * 100.0
    print(f"[k-NN] Accuracy: {acc:.2f}% ({len(test_labels)} samples)")
# (end k-NN code)

# -------------
# Training loop (MODIFIED for Momentum and Dual Views)
# -------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_epoch = 0
    loss_history = []
    resume_data = None

    if args.resume_ckpt is not None:
        print("[RESUME] Loading checkpoint:", args.resume_ckpt)
        resume_data = torch.load(args.resume_ckpt, map_location='cpu')

        # Will load states later after constructing models
        start_epoch = resume_data.get('epoch', -1) + 1
        loss_history = resume_data.get('loss_history', [])
        print(f"[RESUME] Starting from epoch {start_epoch}")

    print("Device:", device)
    
    # Define momentum rate for the teacher EMA update
    # If momentum_rate=0, teacher is frozen (traditional KD)
    # If momentum_rate>0, teacher updates via EMA (BYOL/DINO-style)
    MOMENTUM_RATE = args.teacher_momentum if hasattr(args, 'teacher_momentum') and args.teacher_momentum >= 0 else 0.999

    ds = build_dataset(data_dir=args.data_dir)
    # The loader now yields two views: (v1, v2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)

    # -----------------------------
    # Create teacher and student with DIFFERENT seeds
    # -----------------------------
    IMG_SIZE = 96 # ViT size
    
    # Teacher
    torch.manual_seed(args.teacher_seed)
    if args.use_timm_teacher:
        if timm is None:
            raise RuntimeError("timm requested for teacher but not installed.")
        teacher = ModelWrapper(use_timm=True, timm_name=args.teacher_model, proj_dim=args.proj_dim, img_size=IMG_SIZE).to(device)
    else:
        teacher = ModelWrapper(use_timm=False, proj_dim=args.proj_dim, img_size=IMG_SIZE).to(device)

    # Optionally load teacher checkpoint or pretrained backbone
    if args.teacher_ckpt is not None:
        print("[INFO] Loading teacher checkpoint:", args.teacher_ckpt)
        ck = torch.load(args.teacher_ckpt, map_location='cpu')
        if 'state_dict' in ck:
            teacher.load_state_dict(ck['state_dict'], strict=False)
        elif 'teacher_state' in ck:
            teacher.load_state_dict(ck['teacher_state'], strict=False)
        else:
            teacher.load_state_dict(ck, strict=False)
            
    if args.teacher_pretrained:
        if not args.use_timm_teacher:
            print("[WARN] --teacher_pretrained requires --use_timm_teacher; ignoring pretrained flag.")
        else:
            print("[INFO] Loading timm pretrained weights into teacher backbone (head kept as random).")
            # 💡 FIX: Added img_size=IMG_SIZE to load/interpolate pretrained weights correctly
            t_pre = timm.create_model(args.teacher_model, pretrained=True, num_classes=0, img_size=IMG_SIZE)
            teacher.backbone.load_state_dict(t_pre.state_dict(), strict=False)
            print("[INFO] Teacher backbone loaded from timm pretrained.")

    # Teacher setup: frozen for gradients, but may be updated via EMA
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False  # No gradients flow to teacher
    
    # Print teacher update mode
    if MOMENTUM_RATE > 0:
        print(f"[INFO] Teacher will be updated via EMA (momentum={MOMENTUM_RATE:.4f}) - BYOL/DINO-style")
        if MOMENTUM_RATE >= 0.999:
            print(f"  ⚠️  WARNING: High momentum ({MOMENTUM_RATE:.4f}) may be too conservative for domain adaptation. Consider 0.99 or 0.95.")
    else:
        print("[INFO] Teacher is frozen (traditional KD mode)")

    # Student (different seed)
    torch.manual_seed(args.student_seed)
    if args.use_timm_student:
        if timm is None:
            raise RuntimeError("timm requested for student but not installed.")
        student = ModelWrapper(use_timm=True, timm_name=args.student_model, proj_dim=args.proj_dim, img_size=IMG_SIZE).to(device)
    else:
        student = ModelWrapper(use_timm=False, proj_dim=args.proj_dim, img_size=IMG_SIZE).to(device)


    # Optional classifier for supervised CE (not required for unlabeled KD)
    classifier = None
    if args.supervised_ce:
        if hasattr(ds, 'classes'):
            num_classes = len(ds.classes)
            classifier = nn.Linear(args.proj_dim, num_classes).to(device)
            print("[INFO] Created classifier for supervised CE with num_classes =", num_classes)
        else:
            print("[WARN] supervised_ce requested but dataset has no classes -> ignoring supervised_ce")
            args.supervised_ce = False

    # Optimizers
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if classifier is not None:
        clf_opt = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        clf_opt = None

    # Learning rate schedule: warmup + cosine decay
    warmup_epochs = args.warmup_epochs if hasattr(args, 'warmup_epochs') else 0
    
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    # Use LambdaLR for warmup + cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------- Resume Training Logic --------
    if resume_data is not None:
        print("[RESUME] Restoring model/optimizer/scheduler state...")

        if 'student_state' in resume_data:
            student.load_state_dict(resume_data['student_state'], strict=False)
        if 'teacher_state' in resume_data:
            teacher.load_state_dict(resume_data['teacher_state'], strict=False)
        if 'optimizer' in resume_data:
            optimizer.load_state_dict(resume_data['optimizer'])
        if 'scheduler' in resume_data:
            scheduler.load_state_dict(resume_data['scheduler'])

        print("[RESUME] Resume successful.")



    if clf_opt is not None:
        clf_scheduler = torch.optim.lr_scheduler.LambdaLR(clf_opt, lr_lambda)
    else:
        clf_scheduler = None

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Loss tracking
    loss_history = []
    best_loss = float('inf')

    # training
    num_steps_per_epoch = len(loader)
    global_step = 0
    print(f"[TRAIN] proj_dim={args.proj_dim} batch_size={args.batch_size} epochs={args.epochs} steps/epoch={num_steps_per_epoch}")
    for epoch in range(start_epoch, args.epochs):

        student.train()
        epoch_loss = 0.0
        t0 = time.time()
        # 💡 CHANGE: DataLoader now yields two views: v1 and v2
        for it, ((v1, v2), labels) in enumerate(loader): 
            # Use v1 for teacher, v2 for student (or vice versa)
            v1 = v1.to(device)
            v2 = v2.to(device)
            labels = labels.to(device)

            # teacher forward (no grad) - Use v1
            with torch.no_grad():
                t_feat, t_proj = teacher(v1)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # student forward - Use v2
                s_feat, s_proj = student(v2) 

                # Contrastive KD loss
                loss_contrastive = torch.tensor(0.0, device=device)
                if args.contrastive:
                    # Loss computed between s_proj (from v2) and t_proj (from v1)
                    loss_contrastive = contrastive_loss(s_proj, t_proj, T=args.kd_temp, symmetric=args.symmetric_loss)

                # Total loss
                loss = args.contrastive_weight * loss_contrastive

                # supervised CE (if requested)
                loss_ce = torch.tensor(0.0, device=device)
                if args.supervised_ce and classifier is not None:
                    logits_cls = classifier(s_proj.detach() if args.detach_proj_for_classifier else s_proj)
                    loss_ce = F.cross_entropy(logits_cls, labels)
                    # When supervised, use 'alpha' to balance the two losses
                    loss = (1.0 - args.alpha) * loss_ce + args.alpha * loss 

            # gradient accumulation
            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                # gradient clipping (critical for stability)
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                if clf_opt is not None:
                    scaler.step(clf_opt)
                scaler.update()
                
                optimizer.zero_grad()
                if clf_opt is not None:
                    clf_opt.zero_grad()
                    
                # Update the teacher using momentum after student step
                # If MOMENTUM_RATE=0, teacher stays frozen (traditional KD)
                if MOMENTUM_RATE > 0:
                    update_teacher_momentum(student, teacher, MOMENTUM_RATE)

            epoch_loss += loss.item() * args.grad_accum_steps
            global_step += 1

            if global_step % args.print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # Monitor representation collapse: check feature variance
                with torch.no_grad():
                    feat_var = s_proj.var(dim=0).mean().item()
                    feat_norm = s_proj.norm(dim=1).mean().item()
                print(f"[E{epoch}] step {global_step} loss={loss.item()*args.grad_accum_steps:.4f} contrastive={loss_contrastive.item():.4f} ce={loss_ce.item():.4f} lr={current_lr:.6f} feat_var={feat_var:.4f} feat_norm={feat_norm:.4f}")

        # Update learning rate at end of epoch
        scheduler.step()
        if clf_scheduler is not None:
            clf_scheduler.step()
        
        t1 = time.time()
        avg_loss = epoch_loss / num_steps_per_epoch
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check for improvement
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        # Warning if loss is increasing (potential collapse or instability)
        if len(loss_history) >= 3:
            recent_trend = loss_history[-1] - loss_history[-3]
            if recent_trend > 0.1:  # Loss increased significantly
                print(f"  ⚠️  WARNING: Loss increased by {recent_trend:.4f} over last 3 epochs - possible instability!")
        
        print(f"[E{epoch}] epoch done in {t1-t0:.1f}s avg_loss={avg_loss:.4f} lr={current_lr:.6f} {'(BEST)' if is_best else ''}")
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'student_state': student.state_dict(),
            'teacher_state': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
            'loss_history': loss_history,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.out_dir, f'student_epoch_{epoch}.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(args.out_dir, 'student_best.pth'))
            print(f"  → Saved best model (loss={avg_loss:.4f})")
        
        if classifier is not None:
            torch.save({'classifier_state': classifier.state_dict()}, os.path.join(args.out_dir, f'classifier_epoch_{epoch}.pth'))

    # final save
    torch.save({'student_state': student.state_dict()}, os.path.join(args.out_dir, 'student_final.pth'))
    print("[TRAIN] Saved student_final.pth to", args.out_dir)

    # optional k-NN evaluation
    if args.eval_public_dir:
        knn_eval(student, args.eval_public_dir, device, args)

# -------------
# CLI (MODIFIED for Contrastive Loss arguments)
# -------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default=None, help='ImageFolder folder for training (required)')
    p.add_argument('--out_dir', type=str, default='contrastive_kd_out', help='where to save checkpoints')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--lr', type=float, default=5e-5, help='base learning rate (lowered for stability)')
    p.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs for LR schedule')
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--proj_dim', type=int, default=1024, help='projection/logit dim (1024 recommended for 16GB GPU)')
    
    # Contrastive Loss Arguments
    p.add_argument('--kd_temp', type=float, default=0.07, help='temperature for Contrastive Loss (typical is 0.07)')
    p.add_argument('--alpha', type=float, default=0.5, help='weight for total KD terms vs CE (when supervised CE enabled)')
    p.add_argument('--contrastive', action='store_true', help='enable Contrastive Loss distillation')
    p.add_argument('--contrastive_weight', type=float, default=1.0, help='weight for the contrastive loss term')
    p.add_argument('--symmetric_loss', action='store_true', default=True, help='use symmetric InfoNCE loss (default: True)')
    p.add_argument('--no_symmetric_loss', dest='symmetric_loss', action='store_false', help='use asymmetric loss only')

    # Retained, but less relevant for a pure CKD objective
    p.add_argument('--supervised_ce', action='store_true', help='include supervised CE (requires labels in ImageFolder)')
    p.add_argument('--detach_proj_for_classifier', action='store_true', help='detach proj when training classifier')
    
    # Model/Training args
    p.add_argument('--use_timm_teacher', action='store_true', help='use timm model for teacher')
    p.add_argument('--teacher_model', type=str, default='vit_base_patch16_96', help='timm model name (use _96 suffix for 96x96 images)')
    p.add_argument('--teacher_pretrained', action='store_true', help='load timm pretrained weights into teacher (check assignment rules first)')
    p.add_argument('--teacher_ckpt', type=str, default=None, help='optionally load teacher checkpoint')
    p.add_argument('--use_timm_student', action='store_true', help='use timm model for student')
    p.add_argument('--student_model', type=str, default='vit_small_patch16_96', help='timm model name (use _96 suffix for 96x96 images)')
    p.add_argument('--teacher_seed', type=int, default=0, help='seed for teacher initialization (must differ from student_seed)')
    p.add_argument('--student_seed', type=int, default=1, help='seed for student initialization')
    p.add_argument('--teacher_momentum', type=float, default=0.99, help='momentum rate for teacher EMA update (0.99 for domain adaptation, 0.999 for from-scratch, 0.0 for frozen teacher/traditional KD)')
    p.add_argument('--amp', action='store_true', help='use mixed precision')
    p.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation to emulate larger batch sizes')
    p.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping max norm (0 to disable)')
    p.add_argument('--print_freq', type=int, default=200)
    p.add_argument('--eval_public_dir', type=str, default=None, help='path to eval_public folder (contains train/ and test/) for k-NN eval')
    p.add_argument('--knn_k', type=int, default=50)
    p.add_argument('--resume_ckpt', type=str, default=None,
               help='Path to checkpoint to resume training from')

    return p.parse_args()



def save_loss_plot(loss_history, out_dir):
    """Saves/updates a Plotly HTML file that auto-refreshes every 2 seconds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines+markers',
        name='Epoch Loss'
    ))
    
    fig.update_layout(
        title="Training Loss Per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_dark"
    )

    html_path = os.path.join(out_dir, "loss_curve.html")

    # Wrap Plotly HTML inside an auto-refresh meta tag
    html_content = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
    </head>
    <body>
        {fig.to_html(include_plotlyjs='cdn', full_html=False)}
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html_content)




def main():
    args = parse_args()
    if args.data_dir is None:
        raise RuntimeError("Please provide --data_dir (ImageFolder)")
    if args.teacher_pretrained:
        print("[WARN] --teacher_pretrained enabled. Make sure this complies with your assignment rules (pretrained weights may be disallowed).")
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)

if __name__ == '__main__':
    main()
