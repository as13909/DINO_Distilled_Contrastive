"""
Create Kaggle Submission using YOUR Distilled Student Encoder
==============================================================

This script mirrors the structure of the official KNN baseline,
but replaces the feature extractor with your trained STUDENT model
from DINO_distilled_contrastive.py.

Usage:
    python create_submission_knn_student.py \
        --data_dir ./kaggle_data \
        --ckpt kd_out/student_final.pth \
        --output submission.csv \
        --k 5 \
        --use_timm_student \
        --student_model vit_small_patch16_96
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse

# Import your model classes from the training file
from DINO_distilled_contrastive import ModelWrapper

# =====================================================================
# Feature Extractor using YOUR TRAINED STUDENT MODEL
# =====================================================================

class StudentFeatureExtractor:
    """
    Wraps the trained student model into a simple feature extractor.
    Only the backbone embeddings are used (NOT the projection head).
    """

    def __init__(self, ckpt_path, use_timm, student_model, proj_dim=1024,
                img_size=96, backbone_out=384, device='cuda'):
        self.device = device

        print(f"\nLoading student model: {student_model}")
        self.model = ModelWrapper(
            use_timm=use_timm,
            timm_name=student_model if use_timm else None,
            proj_dim=proj_dim,
            img_size=img_size,
            #backbone_out=backbone_out   # <--- pass correct feature dim
        ).to(device)


        # Load checkpoint
        print(f"Loading checkpoint: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=self.device)

        # Student weights always stored under ["student_state"]
        if "student_state" in ck:
            self.model.load_state_dict(ck["student_state"], strict=False)
        elif "state_dict" in ck:
            self.model.load_state_dict(ck["state_dict"], strict=False)
        else:
            self.model.load_state_dict(ck, strict=False)

        self.model.eval()

        # IMPORTANT: use only the backbone for Kaggle features
        print("  Using backbone embeddings only (not projection head).")

        # Preprocessing: MATCH TRAINING NORMALIZATION
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, pil_image):
        img = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Only use the backbone's pooled feature (first return value)
            feat, _ = self.model(img)
            feat = feat.flatten()

        return feat.cpu().numpy()

    def extract_batch(self, images):
        imgs = [self.transform(im) for im in images]
        imgs = torch.stack(imgs).to(self.device)

        with torch.no_grad():
            # Only use the backbone's pooled feature (first return value)
            feat, _ = self.model(imgs)
            feat = feat

        return feat.cpu().numpy()


# =====================================================================
# Dataset class (simple image loading)
# =====================================================================

class ImageDataset(Dataset):
    def __init__(self, img_dir, filenames, labels=None, resolution=96):
        self.img_dir = Path(img_dir)
        self.filenames = filenames
        self.labels = labels
        self.resolution = resolution

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = self.img_dir / fname

        img = Image.open(path).convert('RGB')

        if self.labels is None:
            return img, fname
        else:
            return img, self.labels[idx], fname


def collate_fn(batch):
    if len(batch[0]) == 3:
        imgs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        fnames = [b[2] for b in batch]
        return imgs, labels, fnames
    else:
        imgs = [b[0] for b in batch]
        fnames = [b[1] for b in batch]
        return imgs, fnames


# =====================================================================
# Feature Extraction Loop
# =====================================================================

def extract_features(loader, fx, split="train"):
    all_feats = []
    all_labels = []
    all_fnames = []

    print(f"\nExtracting {split} features...")

    for batch in tqdm(loader, desc=f"{split}"):
        if len(batch) == 3:
            images, labels, fnames = batch
            labels = list(labels)
        else:
            images, fnames = batch
            labels = None

        feats = fx.extract_batch(images)
        all_feats.append(feats)
        all_fnames.extend(fnames)
        
        if labels is not None:
            all_labels.extend(labels)

    feats = np.concatenate(all_feats, axis=0)
    labels = all_labels if all_labels else None

    print(f"  -> {feats.shape[0]} samples, dim={feats.shape[1]}")
    return feats, labels, all_fnames

# =====================================================================
# Feature Summary Stats (NEW FUNCTION)
# =====================================================================
def print_feature_stats(features, split_name):
    """Prints summary statistics for the extracted feature set."""
    
    # Calculate mean and standard deviation of all feature magnitudes
    mean_val = np.mean(features)
    std_val = np.std(features)
    
    # Calculate the average L2-norm (magnitude) of the vectors
    # L2-norm for each vector (row-wise)
    norms = np.linalg.norm(features, axis=1)
    mean_norm = np.mean(norms)
    
    print(f"\nFeature Stats for {split_name.upper()} (Dim {features.shape[1]}):")
    print("-" * 35)
    print(f"  Mean Feature Value: {mean_val:.4f}")
    print(f"  Std. Dev. of Features: {std_val:.4f}")
    print(f"  Average L2-Norm: {mean_norm:.4f}")
    print("-" * 35)


# =====================================================================
# Train KNN
# =====================================================================

def train_knn(train_feats, train_labels, val_feats, val_labels, k):
    print(f"\nTraining KNN (k={k})...")

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='cosine',
        weights='distance',
        n_jobs=-1
    )

    knn.fit(train_feats, train_labels)

    tr_acc = knn.score(train_feats, train_labels)
    va_acc = knn.score(val_feats, val_labels)
    
    print(f"  Train Acc: {tr_acc:.4f}")
    print(f"  Val Acc:   {va_acc:.4f}")

    return knn


# =====================================================================
# Linear Probe Evaluation (NEW FUNCTION)
# =====================================================================
def run_linear_probe(train_features, train_labels, val_features, val_labels, device="cuda", max_samples=None):
    """
    Train a PyTorch linear probe on frozen features.
    
    Args:
        train_features: numpy array (N_train, feat_dim)
        train_labels: list or numpy array (N_train,)
        val_features: numpy array (N_val, feat_dim)
        val_labels: list or numpy array (N_val,)
        device: torch device
        max_samples: if not None, subsample training data (for speed)
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Linear Probe Evaluation")
    print(f"{'='*60}")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features).to(device)
    train_labels = torch.LongTensor(np.array(train_labels)).to(device)
    val_features = torch.FloatTensor(val_features).to(device)
    val_labels = torch.LongTensor(np.array(val_labels)).to(device)
    
    # Subsample if needed
    if max_samples is not None and len(train_features) > max_samples:
        idx = torch.randperm(len(train_features))[:max_samples]
        train_features = train_features[idx]
        train_labels = train_labels[idx]
        print(f"  Subsampled train from {len(train_features) + len(idx) - max_samples} to {max_samples}")
    
    feat_dim = train_features.shape[1]
    # Assuming labels are 0-indexed and continuous
    num_classes = len(torch.unique(train_labels))
    
    # Create linear classifier
    linear = torch.nn.Linear(feat_dim, num_classes).to(device)
    
    # NOTE: These hyperparameters should be tuned to prevent overfitting.
    # The default values below (weight_decay=1e-3, lr=1e-3) are UNREGULARIZED
    # and likely to overfit the features!
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    batch_size = 128
    num_epochs = 700
    
    linear.train()
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = torch.randperm(len(train_features))
        train_features_shuffled = train_features[perm]
        train_labels_shuffled = train_labels[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_features_shuffled), batch_size):
            batch_features = train_features_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = linear(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Print every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}")
    
    # Evaluation
    linear.eval()
    with torch.no_grad():
        # Train accuracy
        train_outputs = linear(train_features)
        train_preds = train_outputs.argmax(dim=1)
        train_acc = (train_preds == train_labels).float().mean().item()
        
        # Val accuracy
        val_outputs = linear(val_features)
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()
    
    print(f"  Linear probe train acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Linear probe val acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return linear, train_acc, val_acc


# =====================================================================
# Submission creation
# =====================================================================

def create_submission(knn, test_feats, test_names, output):
    print("\nPredicting test labels...")
    preds = knn.predict(test_feats)

    df = pd.DataFrame({
        "id": test_names,
        "class_id": preds
    })
    df.to_csv(output, index=False)

    print(f"\nSaved submission to: {output}")
    print(df.head())


def create_submission_linear_probe(linear, test_feats, test_names, output, device="cuda"):
    print("\nPredicting test labels with LINEAR PROBE...")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    linear.eval()
    with torch.no_grad():
        feats = torch.FloatTensor(test_feats).to(device)
        outputs = linear(feats)
        preds = outputs.argmax(dim=1).cpu().numpy()

    df = pd.DataFrame({
        "id": test_names,
        "class_id": preds
    })

    df.to_csv(output, index=False)
    print(f"\nSaved LINEAR PROBE submission to: {output}")
    print(df.head())



# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--output", default="submission.csv", type=str)
    ap.add_argument("--use_timm_student", action="store_true")
    ap.add_argument("--student_model", type=str, default="vit_small_patch16_224")
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--backbone_out", type=int, default=384,
                help="Feature dimension of the student model backbone")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    test_df = pd.read_csv(data_dir / "test_images.csv")

    # Build datasets
    train_ds = ImageDataset(data_dir / "train",
                            train_df.filename.tolist(),
                            train_df.class_id.tolist(),
                            resolution=args.resolution)

    val_ds = ImageDataset(data_dir / "val",
                          val_df.filename.tolist(),
                          val_df.class_id.tolist(),
                          resolution=args.resolution)

    test_ds = ImageDataset(data_dir / "test",
                           test_df.filename.tolist(),
                           labels=None,
                           resolution=args.resolution)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    # Load your student model
    fx = StudentFeatureExtractor(
    ckpt_path=args.ckpt,
    use_timm=args.use_timm_student,
    student_model=args.student_model,
    proj_dim=args.proj_dim,
    img_size=args.resolution,
    backbone_out=args.backbone_out,  # <--- use correct backbone_out
    device=device
    )


    # Extract features
    tr_feats, tr_labels, _ = extract_features(train_loader, fx, "train")
    va_feats, va_labels, _ = extract_features(val_loader, fx, "val")
    te_feats, _, te_names = extract_features(test_loader, fx, "test")

    # === NEW: Print Feature Stats ===
    print_feature_stats(tr_feats, "train")
    print_feature_stats(va_feats, "val")
    # ================================

    # === Linear Probe Evaluation ===
    # === Train Linear Probe ===
    linear, lp_train_acc, lp_val_acc = run_linear_probe(
        tr_feats, tr_labels, va_feats, va_labels,
        device=device
    )

    # === Create Kaggle submission using LINEAR PROBE ===
    create_submission_linear_probe(
        linear=linear,
        test_feats=te_feats,
        test_names=te_names,
        output=args.output,
        device=device
    )


    print("\nDone! Upload submission.csv to Kaggle.")


if __name__ == "__main__":
    main()