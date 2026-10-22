"""
PyTorch CRNN (CNN + Bidirectional LSTM + CTC Loss) for Handwritten Text Recognition (HTR).
Complete standalone training script runnable locally on CPU or GPU.
"""

import os
import glob
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------
# 1. Vocabulary & Encoding
# -----------------
# Blank token for CTC is at index 0
CHARACTERS = ["<BLANK>"] + sorted(list(string.ascii_letters + string.digits + " .,!?-#':;()\"/$"))
CHAR2IDX = {c: i for i, c in enumerate(CHARACTERS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS)

def encode_text(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

def decode_prediction(pred_indices):
    """Greedy CTC decoder collapsing repeats and blank tokens."""
    res = []
    prev = None
    for idx in pred_indices:
        if idx != prev and idx != 0:
            res.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(res)

# -----------------
# 2. Dataset
# -----------------
class HTRDataset(Dataset):
    def __init__(self, data_dir, img_height=32, max_width=400):
        self.img_height = img_height
        self.max_width = max_width
        self.samples = []

        img_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")) + glob.glob(os.path.join(data_dir, "*.tif")))
        for p in img_paths:
            base = os.path.splitext(p)[0]
            gt_path = f"{base}.gt.txt"
            if os.path.exists(gt_path):
                with open(gt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    self.samples.append((p, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("L")  # Grayscale
        
        # Resize maintaining aspect ratio
        w, h = img.size
        aspect = w / h
        new_w = min(int(self.img_height * aspect), self.max_width)
        img = img.resize((new_w, self.img_height), Image.Resampling.BILINEAR)

        # Pad to max_width
        padded_img = Image.new("L", (self.max_width, self.img_height), color=255)
        padded_img.paste(img, (0, 0))

        # Normalize to [-1, 1]
        tensor = torch.tensor(np.array(padded_img), dtype=torch.float32) / 127.5 - 1.0
        tensor = tensor.unsqueeze(0)  # Shape: (1, H, W)

        encoded = torch.tensor(encode_text(text), dtype=torch.long)
        return tensor, encoded, text

def htr_collate_fn(batch):
    images, targets, raw_texts = zip(*batch)
    images = torch.stack(images, dim=0)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths, raw_texts

# -----------------
# 3. CRNN Architecture
# -----------------
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=128):
        super(CRNN, self).__init__()
        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (H: 32 -> 16, W: 400 -> 200)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (H: 16 -> 8, W: 200 -> 100)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (H: 8 -> 4, W: 100 -> 100)

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((4, 1), (4, 1))   # (H: 4 -> 1, W: 100 -> 100)
        )
        # Sequence Modeling: Bidirectional LSTM
        self.rnn = nn.LSTM(128, hidden_size, bidirectional=True, num_layers=2, batch_first=False)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # (Batch, Channels=128, H=1, W=100)
        features = features.squeeze(2)  # (Batch, 128, 100)
        features = features.permute(2, 0, 1)  # (Seq_len=100, Batch, 128)
        
        rnn_out, _ = self.rnn(features)  # (Seq_len, Batch, hidden*2)
        logits = self.fc(rnn_out)  # (Seq_len, Batch, num_classes)
        return logits

# -----------------
# 4. Training Loop
# -----------------
def train(data_dir, epochs=10, batch_size=8, lr=0.001, save_path="trained_htr_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HTRDataset(data_dir)
    print(f"Loaded {len(dataset)} valid samples from {data_dir}")
    if len(dataset) == 0:
        print("No samples found! Please check data directory.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=htr_collate_fn)

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets, target_lengths, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)  # (T, N, C)
            log_probs = logits.log_softmax(2)

            # Input lengths (constant width sequence after CNN = 100)
            T, N, _ = logits.size()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": CHARACTERS,
        "char2idx": CHAR2IDX
    }, save_path)
    print(f"Model saved successfully to: {save_path}")

    # Evaluate one test sample
    model.eval()
    test_img, _, raw_gt = dataset[0]
    with torch.no_grad():
        test_in = test_img.unsqueeze(0).to(device)
        logits = model(test_in)
        preds = logits.argmax(2).squeeze(1).cpu().numpy()
        pred_text = decode_prediction(preds)

    print("\n--- Test Inference on Sample ---")
    print(f"Ground Truth: {raw_gt}")
    print(f"Predicted:    {pred_text}")
    print("--------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"C:\Users\Hi\.gemini\antigravity\scratch\tesseract_htr_finetune\sample_test_data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--save", type=str, default=r"C:\Users\Hi\.gemini\antigravity\scratch\tesseract_htr_finetune\trained_htr_model.pth")
    args = parser.parse_args()

    train(data_dir=args.data, epochs=args.epochs, save_path=args.save)
