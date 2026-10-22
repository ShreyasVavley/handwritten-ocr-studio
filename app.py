"""
ScribeCraft Pro: Flask Server with:
- Multi-file batch processing & multi-page PDF ingestion
- Automatic Image Enhancement (contrast, deskew, binarization)
- Multi-format Export (Word .docx, Searchable PDF, CSV, TXT)
"""

import os
import glob
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import easyocr
import pypdfium2 as pdfium

from enhancement import enhance_image
from exporters import export_to_txt, export_to_docx, export_to_pdf, export_to_csv

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_data")

print("Initializing EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR Engine initialized and ready!")

def process_single_pil_image(pil_img, deskew=True, contrast=True, binarize=False):
    """Enhance and run OCR on a PIL image."""
    enhanced_pil = enhance_image(pil_img, deskew=deskew, enhance_contrast=contrast, binarize=binarize)
    np_img = np.array(enhanced_pil)
    results = reader.readtext(np_img, detail=0)
    return " ".join(results).strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Accepts multiple uploaded files (images or multi-page PDFs),
    applies selected enhancement toggles, and transcribes them page-by-page.
    """
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    deskew = request.form.get("deskew", "true").lower() == "true"
    contrast = request.form.get("contrast", "true").lower() == "true"
    binarize = request.form.get("binarize", "false").lower() == "true"

    batch_results = []
    full_text_blocks = []

    for file_storage in files:
        filename = file_storage.filename
        file_bytes = file_storage.read()

        if filename.lower().endswith(".pdf"):
            # Multi-page PDF handling via pdfium
            try:
                pdf = pdfium.PdfDocument(file_bytes)
                page_count = len(pdf)
                for page_idx in range(page_count):
                    page = pdf[page_idx]
                    bitmap = page.render(scale=2.0)
                    pil_img = bitmap.to_pil()
                    text = process_single_pil_image(pil_img, deskew, contrast, binarize)
                    entry_text = text if text else "(No text detected on this page)"
                    batch_results.append({
                        "file": f"{filename} (Page {page_idx + 1}/{page_count})",
                        "text": entry_text
                    })
                    full_text_blocks.append(f"--- {filename} [Page {page_idx + 1}] ---\n{entry_text}\n")
            except Exception as e:
                batch_results.append({
                    "file": filename,
                    "text": f"Error rendering PDF: {e}"
                })
        else:
            # Standard image file
            try:
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                text = process_single_pil_image(pil_img, deskew, contrast, binarize)
                entry_text = text if text else "(No text detected)"
                batch_results.append({
                    "file": filename,
                    "text": entry_text
                })
                full_text_blocks.append(f"--- {filename} ---\n{entry_text}\n")
            except Exception as e:
                batch_results.append({
                    "file": filename,
                    "text": f"Error processing image: {e}"
                })

    combined_text = "\n".join(full_text_blocks)
    return jsonify({
        "results": batch_results,
        "combined_text": combined_text
    })

@app.route("/export", methods=["POST"])
def export_file():
    """Exports transcribed text into Word, PDF, CSV, or TXT format."""
    format_type = request.form.get("format", "txt").lower()
    text_content = request.form.get("text", "")

    if not text_content.strip():
        return jsonify({"error": "No text content provided"}), 400

    if format_type == "docx":
        bio = export_to_docx(text_content)
        return send_file(bio, as_attachment=True, download_name="ScribeCraft_Transcribed.docx", mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    elif format_type == "pdf":
        bio = export_to_pdf(text_content)
        return send_file(bio, as_attachment=True, download_name="ScribeCraft_Transcribed.pdf", mimetype="application/pdf")
    elif format_type == "csv":
        bio = export_to_csv(text_content)
        return send_file(bio, as_attachment=True, download_name="ScribeCraft_Transcribed.csv", mimetype="text/csv")
    else:
        bio = export_to_txt(text_content)
        return send_file(bio, as_attachment=True, download_name="ScribeCraft_Transcribed.txt", mimetype="text/plain")

@app.route("/samples")
def list_samples():
    if not os.path.exists(SAMPLES_DIR):
        return jsonify([])
    pngs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png")))[:5]
    results = [{"label": os.path.basename(p), "url": f"/sample_file/{os.path.basename(p)}"} for p in pngs]
    return jsonify(results)

@app.route("/sample_file/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

if __name__ == "__main__":
    print("Starting ScribeCraft Pro on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
