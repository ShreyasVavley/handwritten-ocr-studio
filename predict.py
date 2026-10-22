"""
Inference script for the custom trained CRNN Handwritten Text Recognition (HTR) model.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from train_htr_model import CRNN, decode_prediction

def predict_handwriting(image_path, model_path="trained_htr_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    vocab = checkpoint["vocab"]
    model = CRNN(num_classes=len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess image
    img = Image.open(image_path).convert("L")
    w, h = img.size
    aspect = w / h
    new_w = min(int(32 * aspect), 400)
    img = img.resize((new_w, 32), Image.Resampling.BILINEAR)

    padded_img = Image.new("L", (400, 32), color=255)
    padded_img.paste(img, (0, 0))

    tensor = torch.tensor(np.array(padded_img), dtype=torch.float32) / 127.5 - 1.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 32, 400)

    with torch.no_grad():
        logits = model(tensor)
        preds = logits.argmax(2).squeeze(1).cpu().numpy()
        decoded_text = decode_prediction(preds)

    return decoded_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to handwritten line image (.png/.tif)")
    parser.add_argument("--model", type=str, default="trained_htr_model.pth", help="Path to trained .pth model")
    args = parser.parse_args()

    recognized_text = predict_handwriting(args.image, args.model)
    print(f"Recognized Text: {recognized_text}")
