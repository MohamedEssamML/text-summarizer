# 📝 Text Summarization with BART

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-green)

**A state-of-the-art NLP application for extractive and abstractive text summarization using Facebook's BART model.**

## ✨ Key Features
- **Dual-Mode Summarization**  
  ✅ *Extractive* (key sentences) & ✅ *Abstractive* (paraphrased content)  
- **Smart Length Control**  
  📏 Choose short (25%), medium (50%), or long (75%) summaries  
- **ROUGE Metrics**  
  📊 Automatic quality evaluation when reference summaries are provided  
- **File & Text Input**  
  📂 Supports `.txt`, `.pdf`, and direct text input  
- **User-Friendly UI**  
  🎨 Beautiful Streamlit interface with dark/light mode  

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/MohamedEssamML/text-summarizer.git
cd text-summarizer
pip install -r requirements.txt
