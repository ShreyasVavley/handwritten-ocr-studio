# ScribeCraft • Handwritten Text Recognition (HTR) Studio

An AI-powered Handwritten Text Recognition application featuring an interactive monochrome Web UI and deep learning OCR engine.

---

## Features
- **Deep Vision OCR**: Powered by deep neural text detection (CRAFT) and recognition (ResNet + LSTM).
- **Interactive Monochrome Web Studio**: Clean black & white interface with drag-and-drop image upload, instant transcription, and clipboard copy.
- **Pre-loaded Samples**: Test handwriting recognition with one click.
- **Custom Training Pipeline**: Includes scripts to generate synthetic cursive handwriting datasets and fine-tune custom CRNN/CTC or Tesseract 5 LSTM models.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## Project Structure
```
├── app.py                     # Flask web app & EasyOCR inference server
├── templates/
│   └── index.html             # Minimalist black & white Web UI
├── sample_test_data/          # Sample handwriting images for instant testing
├── generate_synthetic_data.py # Synthetic cursive handwriting generator
├── train_htr_model.py         # PyTorch CRNN + CTC model definition & training
├── predict.py                 # CLI inference script
├── Tesseract_HTR_FineTuning.ipynb # Google Colab fine-tuning notebook
├── requirements.txt
└── README.md
```
