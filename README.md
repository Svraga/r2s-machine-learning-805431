# r2s-machine-learning-805431
Repository for a Machine Learning project on NLP

# Multilingual Sentence Similarity Analyzer

A simple web application that calculates the semantic similarity between two sentences, supporting multiple languages.

## Quick Setup Guide

### Required Files
You only need these 3 files in the same directory:
- [`app.py`](https://github.com/svraga/r2s-machine-learning805431/raw/main/app.py) - The main application file
- [`bert_similarity_model_weights.pth`](https://github.com/svraga/r2s-machine-learning805431/raw/main/weights/bert_similarity_model_weights.pth) - Model weights (download from weights folder)
- [`requirements.txt`](https://github.com/svraga/r2s-machine-learning805431/raw/main/requirements.txt) - Python dependencies

### Installation and Setup

1. **Download the necessary files** (links above)
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
4. **Use the application**
   - Open your web browser and go to: http://127.0.0.1:5000/
   - Enter two sentences (they can be in different languages)
   - Click "Calculate Similarity" to get the similarity score
  ## About
- This application uses a fine-tuned multilingual BERT model to calculate semantic similarity between sentences in different languages. 
- **Multilingual Support**: Works with the following languages:  
  Italian | English | French | Spanish | German  
  Portuguese | Dutch | Japanese | Russian | Polish | Chinese
- The model outputs a score from 0 to 5, where:0 No similarity, 5 Nearly identical meaning

  
