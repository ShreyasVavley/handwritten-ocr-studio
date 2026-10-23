# ScribeCraft Pro • Handwritten Text Recognition (HTR) Studio

<div align="center">

![CI Status](https://github.com/ShreyasVavley/handwritten-ocr-studio/actions/workflows/ci.yml/badge.svg)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/Web%20App-Flask-black.svg)](https://flask.palletsprojects.com/)
[![OCR Engine: EasyOCR](https://img.shields.io/badge/OCR%20Engine-EasyOCR%20%2B%20PyTorch-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <strong>A high-performance document transcription suite featuring an interactive digital handwriting drawing canvas, multi-page PDF processing, automatic CV enhancements, and a warm Ivory aesthetic.</strong>
</p>

</div>

---

## 🌟 Pro Features & Highlights

### ✍️ Interactive Digital Handwriting Slate
- **Live Canvas Drawing**: Scribble cursive, signatures, or notes directly on the screen using your mouse, trackpad, or touchscreen stylus.
- **Multiple Ink Types & Nib Sizes**: Toggle between Classic Obsidian (`#18181b`), Fountain Pen Blue (`#1d4ed8`), or Forest Green (`#047857`) with adjustable line width.
- **Ruled Paper Guidelines**: Realistic cursive handwriting baseline guides to keep letter proportions straight.
- **Instant Real-Time OCR**: Transcribe your digital strokes in seconds with one click.

### 📑 Multi-Page PDF & Batch Uploads
- Ingest entire multi-page PDF binders, scanned folders, or multiple images simultaneously.
- Sequential rendering via `pypdfium2` with structured, page-by-page feed cards.

### 🪄 Automated Computer Vision Enhancements
- **Auto-Deskew**: Automatically detects rotated, crooked, or tilted handwriting lines and straightens them.
- **Contrast Boost (CLAHE)**: Enhances faint ballpoint ink, pencil notes, and camera shadows.
- **Adaptive Binarization**: Cleans uneven lighting gradients, leaving crisp black text on white paper.

### 💾 Multi-Format Document Exporting
- **📄 Word (.docx)**: Fully formatted Microsoft Word documents.
- **📑 Searchable PDF**: Formatted standard PDF with custom font layout.
- **📊 CSV (.csv)**: Line-numbered tabular output for data analysis and spreadsheets.
- **📝 TXT**: Clean UTF-8 plain text file.

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyasVavley/handwritten-ocr-studio.git
cd handwritten-ocr-studio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Studio
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:5000
```

---

## 🛠️ Project Structure
```
handwritten-ocr-studio/
├── .github/workflows/ci.yml   # Automated GitHub Actions CI test suite
├── app.py                     # Flask server, multi-page PDF engine & EasyOCR inference
├── enhancement.py             # Computer Vision preprocessing (Deskew, CLAHE, Binarization)
├── exporters.py               # Document exporters (.docx, .pdf, .csv, and .txt)
├── templates/
│   └── index.html             # Ivory & Obsidian interface with live drawing canvas
├── sample_test_data/          # Synthetic cursive sample snippets
├── generate_synthetic_data.py # Synthetic cursive dataset generator
├── train_htr_model.py         # Standalone PyTorch CRNN + BiLSTM model trainer
├── predict.py                 # CLI inference script
├── test_client.py             # Automated API testing suite
├── Tesseract_HTR_FineTuning.ipynb # Google Colab notebook for Tesseract 5 fine-tuning
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
