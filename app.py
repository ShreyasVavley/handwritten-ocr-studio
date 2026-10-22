"""
Production HTR Web Application powered by EasyOCR.
Accurately transcribes both handwritten and printed text in real-time.
"""

import os
import glob
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import easyocr

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_data")

print("Initializing EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR Engine initialized and ready!")

def run_inference(image_bytes):
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img)

        results = reader.readtext(np_img, detail=0)
        recognized_text = " ".join(results).strip()

        if not recognized_text:
            return "(No text detected)"
        return recognized_text
    except Exception as e:
        print(f"Error in OCR: {e}")
        return f"Error: {e}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image_bytes = file.read()
    pred_text = run_inference(image_bytes)
    return jsonify({"text": pred_text})

@app.route("/samples")
def list_samples():
    if not os.path.exists(SAMPLES_DIR):
        return jsonify([])
    
    pngs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png")))[:5]
    results = []
    for p in pngs:
        filename = os.path.basename(p)
        results.append({
            "label": filename,
            "url": f"/sample_file/{filename}"
        })
    return jsonify(results)

@app.route("/sample_file/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

if __name__ == "__main__":
    print("Starting HTR Web App on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
