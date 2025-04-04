# LegalNLP: AI-Powered Legal Text Summarization

## 📌 Overview
LegalNLP is a professional-grade Natural Language Processing (NLP) project focused on summarizing legal documents. This project leverages state-of-the-art transformer models to generate concise summaries while preserving key legal information.

## 🚀 Features
- **Legal Text Summarization**: Extracts key points from lengthy legal documents.
- **Hugging Face Model Integration**: Compatible with `transformers` for easy model loading and deployment.

## 🛠 Installation
Clone the repository and set up the virtual environment:
```bash
git clone https://github.com/KushalSuvan/yama.git
cd yama
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Training the Model
Train the summarization model using:
```bash
python src/models/summarizer/train.py
```

## 🔍 Running Inference
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("KushalSuvan/yama_summarizer")
output = model(input_data)
```

## 📤 Deploying to Hugging Face
Upload trained models:
```bash
python push_to_huggingface.py
```

## 🧪 Testing
Run unit tests:
```bash
pytest tests/
```

## 📜 License
This project is licensed under the Apache License 2.0.

## 🙌 Contributors
- **Kushal Suvan Jenamani**
- **Shashank Tippanavar**
- **Danesh Toshniwal**
- **Akshat**


