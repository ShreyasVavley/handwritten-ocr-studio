# ScribeCraft • Handwritten Text Recognition (HTR) Studio

> **An AI-powered document transcription web application designed with a minimalist Ivory aesthetic and powered by deep neural OCR.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%2B%20EasyOCR-black.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Highlights
- **Intelligent Handwriting Recognition**: Accurately recognizes cursive handwriting, doctor notes, signatures, and natural script using deep CRAFT text detection and ResNet-LSTM sequence recognition.
- **Ivory & Obsidian UI**: A serene, distraction-free aesthetic with warm ivory canvas (`#fcfbf9`), crisp cards, smooth drag-and-drop file upload, and one-click clipboard copy.
- **Instant Sample Testing**: Includes pre-loaded synthetic handwriting snippets to test transcription speed and accuracy out of the box.
- **Fine-Tuning Toolchain**: Contains end-to-end pipelines for generating custom cursive datasets and fine-tuning CRNN/CTC or Tesseract 5 LSTM models on Google Colab or cloud GPUs.

---

## 🖼️ UI Preview
The portal features an elegant, humanized interface with:
- Drag-and-drop single-line or cropped handwritten text uploads.
- Instant live image preview and removal controls.
- Fast local inference on CPU or GPU.
- Clean typography pairing **Space Grotesk** and natural **Caveat** script.

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
├── app.py                     # Flask web server & EasyOCR inference engine
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

## 🔬 Model Training & Fine-Tuning

### Synthetic Handwriting Generation
Generate synthetic cursive lines with random variations (rotation, ink bleed, and paper textures):
```bash
python generate_synthetic_data.py --count 150 --out data/my_handwriting-ground-truth
```

### Train Custom CRNN Model
Train a lightweight PyTorch sequence-to-sequence model locally:
```bash
python train_htr_model.py --data sample_test_data --epochs 30
```

### Cloud GPU Fine-Tuning
Upload `Tesseract_HTR_FineTuning.ipynb` to [Google Colab](https://colab.research.google.com/) for high-precision fine-tuning using `tessdata_best`.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
