import os
import random
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# =========================
SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001
WEIGHT_DECAY = 5e-4
VAL_SPLIT = 0.10
TTA_ROUNDS = 20  # Optimized TTA rounds
EARLY_STOP_PATIENCE = 8  # Early stopping patience

TRAIN_PATH = r"/Users/ayesh/Code/fashion-mnist-model/Fashion-MNIST-CNN/train.csv"
TEST_PATH = r"/Users/ayesh/Code/fashion-mnist-model/Fashion-MNIST-CNN/test.csv"

# =========================
# SEED FOR REPRODUCIBILITY
# =========================
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# DEVICE SELECTION (macOS optimized)
# =========================
if torch.cuda.is_available():
    DEVICE = "cuda"
    use_amp = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    use_amp = False  # MPS doesn't support AMP yet
else:
    DEVICE = "cpu"
    use_amp = False

print(f"🚀 Using device: {DEVICE}")
print(f"⚡ Mixed precision: {use_amp}")
print(f"⏱️  Estimated time: ~30-40 minutes\n")

# =========================
# LOAD DATA
# =========================
print("📂 Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

pixel_cols = [c for c in train_df.columns if c != "label"]

X = train_df[pixel_cols].values.astype(np.float32) / 255.0
y = train_df["label"].values.astype(np.int64)

test_ids = test_df["id"].values
X_test = test_df[pixel_cols].values.astype(np.float32) / 255.0

X = X.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y
)

print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")


# =========================
# ENHANCED AUGMENTATION
# =========================
def augment(x):
    """Enhanced augmentation with multiple techniques"""
    x = x.unsqueeze(0) if x.dim() == 3 else x
    
    # Random rotation + translation + scale
    angle = (torch.rand(1, device=x.device) * 2 - 1) * 0.174  # ±10°
    scale = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * 0.1
    tx = (torch.rand(1, device=x.device) * 2 - 1) * 0.1
    ty = (torch.rand(1, device=x.device) * 2 - 1) * 0.1

    cos_a = torch.cos(angle) * scale
    sin_a = torch.sin(angle) * scale

    theta = torch.zeros((x.size(0), 2, 3), device=x.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    x_aug = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Random horizontal flip (50% chance)
    if torch.rand(1).item() < 0.5:
        x_aug = torch.flip(x_aug, dims=[3])

    # Random erasing (Cutout-like)
    if torch.rand(1).item() < 0.3:
        h, w = 28, 28
        erase_h = torch.randint(3, 8, (1,)).item()
        erase_w = torch.randint(3, 8, (1,)).item()
        top = torch.randint(0, max(1, h - erase_h + 1), (1,)).item()
        left = torch.randint(0, max(1, w - erase_w + 1), (1,)).item()
        x_aug[:, :, top:min(h, top + erase_h), left:min(w, left + erase_w)] = 0.0

    # Random brightness/contrast adjustment
    if torch.rand(1).item() < 0.3:
        brightness = 0.8 + torch.rand(1, device=x.device).item() * 0.4
        x_aug = torch.clamp(x_aug * brightness, 0.0, 1.0)

    return x_aug.squeeze(0) if x.dim() == 4 and x.size(0) == 1 else x_aug


# =========================
# DATASET
# =========================
class FashionDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment(x)
        if self.y is None:
            return x
        return x, self.y[idx]


train_loader = DataLoader(
    FashionDataset(X_train, y_train, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, 
    pin_memory=(DEVICE != "cpu")
)

val_loader = DataLoader(
    FashionDataset(X_val, y_val, augment=False),
    batch_size=512, shuffle=False, num_workers=0
)


# =========================
# IMPROVED MODEL ARCHITECTURE
# =========================
class ImprovedCNN(nn.Module):
    """Enhanced CNN with residual connections for Fashion-MNIST"""
    def __init__(self):
        super().__init__()
        
        # Initial stem
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 1: 64 -> 128, 28x28 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.shortcut1 = nn.Conv2d(64, 128, 1, bias=False)
        self.bn_shortcut1 = nn.BatchNorm2d(128)
        
        # Block 2: 128 -> 256, 14x14 -> 7x7
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.shortcut2 = nn.Conv2d(128, 256, 1, bias=False)
        self.bn_shortcut2 = nn.BatchNorm2d(256)
        
        # Block 3: 256 -> 512, 7x7 -> 4x4
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.shortcut3 = nn.Conv2d(256, 512, 1, bias=False)
        self.bn_shortcut3 = nn.BatchNorm2d(512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout2d(x, 0.1, self.training)
        
        # Block 1 with residual
        identity = self.bn_shortcut1(self.shortcut1(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = F.relu(x + identity)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, 0.2, self.training)
        
        # Block 2 with residual
        identity = self.bn_shortcut2(self.shortcut2(x))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = F.relu(x + identity)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, 0.3, self.training)
        
        # Block 3 with residual
        identity = self.bn_shortcut3(self.shortcut3(x))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.bn8(self.conv8(x))
        x = F.relu(x + identity)
        x = F.dropout2d(x, 0.4, self.training)
        
        # Classifier
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        return x


model = ImprovedCNN().to(DEVICE)
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# =========================
# TRAINING SETUP
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    div_factor=10.0,
    final_div_factor=100.0
)

# Mixed precision scaler (only for CUDA)
if use_amp and DEVICE == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None


# =========================
# MIXUP AUGMENTATION
# =========================
def mixup(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


# =========================
# TRAINING LOOP WITH EARLY STOPPING
# =========================
print("🔥 TRAINING START\n")
best_val_acc = 0.0
patience_counter = 0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # Mixup
        xb, ya, yb, lam = mixup(xb, yb, alpha=0.2)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        train_loss += loss.item()
        train_total += xb.size(0)
        train_correct += ((lam * (logits.argmax(1) == ya).float() + 
                          (1 - lam) * (logits.argmax(1) == yb).float()) > 0.5).sum().item()

    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += yb.size(0)

    val_acc = val_correct / val_total
    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1

    # Print progress
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.5f} | "
              f"Val Acc: {val_acc:.5f} | Best: {best_val_acc:.5f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed/60:.1f}m")
    
    # Early stopping
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
        break

print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.5f}\n")

# =========================
# RETRAIN ON FULL DATA
# =========================
print("🔁 Retraining on full data for final boost...")

# Combine train + val
X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)

full_loader = DataLoader(
    FashionDataset(X_full, y_full, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=(DEVICE != "cpu")
)

# Load best weights
model.load_state_dict(torch.load("best_model.pt"))

# Retrain for additional epochs with lower learning rate
optimizer = optim.AdamW(model.parameters(), lr=LR / 3, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

for epoch in range(10):
    model.train()
    for xb, yb in full_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb, ya, yb, lam = mixup(xb, yb, alpha=0.15)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            optimizer.step()
    
    scheduler.step()
    if (epoch + 1) % 3 == 0:
        print(f"  Retrain epoch {epoch + 1}/10 | LR: {scheduler.get_last_lr()[0]:.6f}")

print("✅ Full data retraining complete!\n")

# =========================
# OPTIMIZED TTA PREDICTION
# =========================
print(f"🔮 Predicting with {TTA_ROUNDS}x TTA...")

test_loader = DataLoader(
    FashionDataset(X_test, y=None, augment=False),
    batch_size=256, shuffle=False, num_workers=0
)

model.eval()
all_probs = []

with torch.no_grad():
    for batch_idx, xb in enumerate(test_loader):
        xb = xb.to(DEVICE)
        
        # Collect TTA predictions
        tta_probs = []
        
        # Original prediction (weighted 2x)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        tta_probs.append(probs)
        tta_probs.append(probs)  # Double weight for original
        
        # Augmented predictions
        for _ in range(TTA_ROUNDS - 2):
            # Batch augmentation is more efficient
            xb_aug = torch.stack([augment(img) for img in xb])
            logits = model(xb_aug)
            tta_probs.append(F.softmax(logits, dim=1))
        
        # Average probabilities
        batch_probs = torch.stack(tta_probs).mean(dim=0).cpu().numpy()
        all_probs.append(batch_probs)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

all_probs = np.concatenate(all_probs, axis=0)
final_preds = all_probs.argmax(axis=1)

# =========================
# CREATE SUBMISSION
# =========================
submission = pd.DataFrame({
    "id": test_ids,
    "label": final_preds
})

submission.to_csv("submission.csv", index=False)

print("\n" + "=" * 70)
print("✅ SUBMISSION SAVED: submission.csv")
print(f"📈 Best Validation Accuracy: {best_val_acc:.5f}")
print(f"📊 Test predictions: {len(final_preds)} samples")
print("=" * 70)
print("\n" + submission.head(20).to_string())
print(f"\n✅ All done! Total time: {(time.time() - start_time)/60:.1f} minutes")
