# ScribeCraft • Handwritten Text Recognition (HTR) Studio

<div align="center">

![CI Status](https://github.com/ShreyasVavley/handwritten-ocr-studio/actions/workflows/ci.yml/badge.svg)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flask](https://img.shields.io/badge/Web%20App-Flask-black.svg)](https://flask.palletsprojects.com/)
[![OCR Engine: EasyOCR](https://img.shields.io/badge/OCR%20Engine-EasyOCR%20%2B%20PyTorch-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <strong>An AI-powered document transcription web application designed with a minimalist Ivory aesthetic and powered by deep neural OCR.</strong>
</p>

</div>

---

## 🌟 Highlights
- **Intelligent Handwriting Recognition**: Accurately recognizes cursive handwriting, doctor notes, signatures, and natural script using deep CRAFT text detection and ResNet-LSTM sequence recognition.
- **Ivory & Obsidian UI**: A serene, distraction-free aesthetic with warm ivory canvas (`#fcfbf9`), crisp cards, smooth drag-and-drop file upload, and one-click clipboard copy.
- **Multi-Page PDF & Batch Processing**: Ingest and transcribe entire multi-page PDF documents or image batches in parallel.
- **Automatic Image Enhancement**: Built-in automated deskew, contrast boost (CLAHE), and adaptive binarization filters.
- **Multi-Format Exports**: One-click download as Word (`.docx`), Searchable PDF, Spreadsheet CSV, or plain text.
- **Fine-Tuning Toolchain**: End-to-end pipelines for generating custom synthetic cursive handwriting datasets and fine-tuning CRNN/CTC or Tesseract 5 LSTM models on Google Colab.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyasVavley/handwritten-ocr-studio.git
cd handwritten-ocr-studio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Studio
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🛠️ Project Structure
```
handwritten-ocr-studio/
├── .github/workflows/ci.yml   # Automated GitHub Actions CI workflow
├── app.py                     # Flask web server & EasyOCR inference engine
├── enhancement.py             # Image enhancement (deskew, CLAHE contrast, binarization)
├── exporters.py               # Exporters for .docx, .pdf, .csv, and .txt
├── templates/
│   └── index.html             # Minimalist Ivory & Obsidian web interface
├── sample_test_data/          # Sample handwriting images & ground truth pairs
├── generate_synthetic_data.py # Synthetic cursive handwriting generator
├── train_htr_model.py         # PyTorch CRNN + BiLSTM model training script
├── predict.py                 # Standalone command-line inference tool
├── test_client.py             # Automated API test script
├── Tesseract_HTR_FineTuning.ipynb # Google Colab notebook for Tesseract 5 LSTM
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
