# 🧠 GoEmotions Text Emotion Recognition API

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![FastAPI](https://img.shields.io/badge/FastAPI-API-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A multi-label emotion classification API built with BERT, trained on Google’s GoEmotions dataset, deployed via FastAPI. Predicts 28 different emotions from text input with confidence scores.

---

## ✨ Overview

* 🧠 Fine-tuned BERT for 28 emotion categories  
* 📝 Multi-label classification with sigmoid activation  
* ⚡ CPU-friendly inference API with FastAPI  
* 💻 Clean modular code for training, evaluation, and deployment  
* 📊 Confidence scores for predicted emotions  

---

## 🧠 Supported Emotions

`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral`

---

## 🏗 Project Structure

```
MultiModal-Emotion-Recognition/

├── data/ 
  ├── goemotions_loader.py 
  └── image_dataset.py 
├── models/ 
  └── multimodal_emotion.py 
├── utils/ 
  ├── config.py 
  └── metrics.py 
├── train.py 
├── main.py 
├── requirements.txt 
├── README.md 
├── .gitignore`

```
---

## 📦 Installation

```bash
git clone https://github.com/yourusername/goemotions-api.git
cd goemotions-api
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
🎯 Training
```bach
python train.py
```
* Loads GoEmotions dataset

* Preprocesses text into tokenized tensors

* Fine-tunes BERT for multi-label emotion classification

* Saves best checkpoint locally

⚠️ Checkpoints and TensorBoard logs are ignored from GitHub via .gitignore.

⚡ Run the API
```bash
uvicorn main:app --reload
```
Swagger docs: http://127.0.0.1:8000/docs

📥 Example API Request
POST /predict:
```bash
{"text": "I am so happy today!"}
```
📤 Example API Response
```bash

{"text": "I am so happy today!","predictions":[{"emotion":"joy","confidence":0.9874},{"emotion":"excitement","confidence":0.6817},{"emotion":"optimism","confidence":0.5912}]}
```
🔬 Model Details
* Base Model: BERT (Transformer Encoder)

* Loss Function: Binary Cross Entropy with Logits

* Activation: Sigmoid

* Task Type: Multi-label classification

* Dataset: Google GoEmotions (28 emotions)

🛠 Tech Stack
* PyTorch

* HuggingFace Transformers

* FastAPI

* Uvicorn

* TensorBoard (optional)

💡 How It Works
* Text input is tokenized using a pretrained BERT tokenizer

* Model outputs logits for each of the 28 emotions

* Sigmoid converts logits into probabilities

* Top-K highest confidence emotions are returned

📌 Future Improvements
* Threshold tuning per emotion

* Confidence calibration

* Model quantization for faster CPU inference

* Docker deployment

* Web UI / HuggingFace Spaces demo

📄 License
MIT License

🙌 Author
Built as an applied NLP project focusing on multi-label emotion recognition and production-ready ML deployment.

.gitignore
```bash

# Checkpoints
checkpoints/
*.pt

# TensorBoard logs
runs/

# Python cache
__pycache__/
*.pyc

# Virtual env
venv/
env/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```
requirements.txt
```bash

torch
transformers
fastapi
uvicorn
pydantic
tensorboard
torchvision
```



