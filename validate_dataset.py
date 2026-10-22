import os
from PIL import Image

def validate_dataset(gt_dir: str):
    """
    Validates that each line image (.png or .tif) has a corresponding .gt.txt file
    and matches formatting required by Tesseract LSTM training.
    """
    if not os.path.isdir(gt_dir):
        print(f"Directory not found: {gt_dir}")
        return

    files = os.listdir(gt_dir)
    image_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
    
    valid_count = 0
    missing_gt = 0
    empty_gt = 0

    print(f"Total candidate images found: {len(images)}")

    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        gt_name = f"{base_name}.gt.txt"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            empty_gt += 1
            continue

        valid_count += 1

    print(f"Valid line pairs: {valid_count}")
    print(f"Missing .gt.txt: {missing_gt}")
    print(f"Empty .gt.txt: {empty_gt}")

if __name__ == "__main__":
    target_dir = os.path.join(".", "data", "my_handwriting-ground-truth")
    validate_dataset(target_dir)
