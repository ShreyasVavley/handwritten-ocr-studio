"""
Synthetic Handwritten Line Generator for Tesseract HTR Fine-Tuning
Downloads open-source handwriting Google Fonts and renders synthetic
line images paired with .gt.txt files, with subtle rotation, noise, and scaling.
"""

import os
import random
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_URLS = [
    # Caveat
    ("Caveat-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf"),
    # Shadows Into Light
    ("ShadowsIntoLight.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight.ttf"),
    # Kalam
    ("Kalam-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf")
]

SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Antigravity neural network handwriting recognition model.",
    "Invoice number INV-98234 dated 14th November.",
    "Please send payment confirmation to the accounting department.",
    "Total balance due is $1,250.75 by next Friday.",
    "Handwritten notes on machine learning and optical character recognition.",
    "Doctor prescription: Take one tablet after breakfast and dinner daily.",
    "Meeting rescheduled to 3:30 PM in conference room B.",
    "Package tracking number: 9400 1118 9922 3847 1234 56.",
    "Signature verified and authorized by branch manager.",
    "Customer delivery address: 742 Evergreen Terrace, Springfield.",
    "Laboratory test results pending review by senior pathologist.",
    "Please return this form signed along with a copy of your identification.",
    "Quarterly financial projections show steady growth across all divisions.",
    "Temperature recorded at 24 degrees Celsius with moderate humidity."
]

def download_fonts(fonts_dir="fonts"):
    os.makedirs(fonts_dir, exist_ok=True)
    font_paths = []
    for filename, url in FONT_URLS:
        dest_path = os.path.join(fonts_dir, filename)
        if not os.path.exists(dest_path):
            print(f"Downloading font: {filename}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
            except Exception as e:
                print(f"Warning: Failed downloading {url}: {e}")
                continue
        if os.path.exists(dest_path):
            font_paths.append(dest_path)
    return font_paths

def render_handwritten_line(text, font_path, font_size=32):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text bounding box
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Add padding
    pad_x, pad_y = 30, 20
    img_w = max(text_w + pad_x * 2, 200)
    img_h = max(text_h + pad_y * 2, 50)

    # Background: slight off-white paper texture
    bg_color = (random.randint(245, 255), random.randint(245, 255), random.randint(240, 250))
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    d = ImageDraw.Draw(img)

    # Text color: dark ink (black / dark blue)
    ink_choice = random.choice([
        (random.randint(10, 35), random.randint(10, 35), random.randint(10, 35)),  # Black ink
        (random.randint(10, 25), random.randint(20, 45), random.randint(70, 110))  # Blue ink
    ])

    text_x = pad_x - bbox[0]
    text_y = pad_y - bbox[1]
    d.text((text_x, text_y), text, font=font, fill=ink_choice)

    # Slight rotation (-1.5 to +1.5 degrees) for handwriting realism
    rot_angle = random.uniform(-1.5, 1.5)
    img = img.rotate(rot_angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)

    # Subtle blur to simulate ink bleed
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img

def generate_dataset(output_dir="data/my_handwriting-ground-truth", num_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    font_paths = download_fonts("fonts")

    if not font_paths:
        print("Error: No fonts available. Using system default.")
        font_paths = [None]

    print(f"Generating {num_samples} synthetic handwritten lines into: {output_dir}")

    for idx in range(num_samples):
        text = random.choice(SAMPLE_SENTENCES)
        # Random variations (append random numbers or dates)
        if random.random() < 0.4:
            text = f"{text} #{random.randint(100, 9999)}"

        font_path = random.choice(font_paths)
        font_size = random.randint(28, 38)

        img = render_handwritten_line(text, font_path, font_size=font_size)

        base_name = f"synth_line_{idx:05d}"
        img_path = os.path.join(output_dir, f"{base_name}.png")
        gt_path = os.path.join(output_dir, f"{base_name}.gt.txt")

        # Save image and ground truth text
        img.save(img_path)
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

    print(f"Successfully generated {num_samples} samples!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic handwritten text dataset")
    parser.add_argument("--count", type=int, default=150, help="Number of line samples to generate")
    parser.add_argument("--out", type=str, default="data/my_handwriting-ground-truth", help="Output directory")
    args = parser.parse_args()

    generate_dataset(output_dir=args.out, num_samples=args.count)
