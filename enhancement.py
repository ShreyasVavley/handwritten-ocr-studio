"""
Image Enhancement and Preprocessing module for OCR.
Provides auto-contrast, adaptive binarization, de-skewing, and noise removal.
"""

import cv2
import numpy as np
from PIL import Image

def deskew_image(image_cv):
    """
    Detects skew angle of text lines using minimum area bounding rectangle
    and rotates the image to make text horizontal.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Invert binary to find text contours
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image_cv  # Not enough points to reliably compute angle

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Only deskew if slight tilt (avoid accidental 90 degree flips)
    if abs(angle) > 25:
        return image_cv

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_image(pil_img, deskew=True, enhance_contrast=True, binarize=False):
    """
    Applies image enhancements to optimize handwriting recognition.
    """
    # Convert PIL to OpenCV BGR
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if deskew:
        img_cv = deskew_image(img_cv)

    if enhance_contrast:
        # Convert to LAB color space for CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if binarize:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding for uneven lighting/shadows
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        img_cv = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Convert back to PIL RGB
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
