
from flask import Flask, send_from_directory, request, jsonify
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTSimilarityModel(nn.Module):
    def __init__(self):
        super(BERTSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        x = self.dropout(outputs.pooler_output)
        return self.regressor(x).squeeze(-1)

app = Flask(__name__)

# Verifica se esiste la cartella "static"
static_dir = os.path.join(os.getcwd(), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    print(f"Cartella 'static' creata in {static_dir}")

# Crea un file HTML statico nella cartella "static"
static_html_path = os.path.join(static_dir, "index.html")
with open(static_html_path, "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentence Similarity Checker</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 20px;
        }
        .container {
            max-width: 800px;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        h1 {
            margin-bottom: 25px;
            color: #0d6efd;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .result-container {
            margin-top: 25px;
            padding: 15px;
            border-radius: 5px;
            background-color: #f0f8ff;
            display: none;
        }
        #loadingSpinner {
            display: none;
        }
        footer {
            margin-top: 30px;
            font-size: 0.9em;
            color: #6c757d;
            text-align: center;
        }
        .score-display {
            font-size: 20px;
            font-weight: bold;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Sentence Similarity Analyzer</h1>
        <p class="lead text-center mb-4">Enter two sentences to check their semantic similarity</p>
        
        <form id="similarityForm">
            <div class="form-group">
                <label for="sentence1">Sentence 1:</label>
                <textarea class="form-control" id="sentence1" rows="3" placeholder="Enter the first sentence" required></textarea>
            </div>
            
            <div class="form-group">
                <label for="sentence2">Sentence 2:</label>
                <textarea class="form-control" id="sentence2" rows="3" placeholder="Enter the second sentence" required></textarea>
            </div>
            
            <div class="text-center">
                <button type="submit" class="btn btn-primary btn-lg">
                    Calculate Similarity
                </button>
                <div class="mt-3" id="loadingSpinner">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Processing sentences...</p>
                </div>
            </div>
        </form>
        
        <div class="result-container" id="resultContainer">
            <h4>Similarity Score: <span id="scoreValue"></span>/5</h4>
            <div class="d-flex align-items-center justify-content-center">
                <div class="progress" style="height: 30px; width: 100%;">
                    <div id="scoreProgressBar" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="5"></div>
                </div>
            </div>
            <div class="score-display">
                <span id="scoreDisplay">0.000</span>/5
            </div>
            <p class="mt-3" id="interpretationText"></p>
        </div>
        
        <footer>
            <p>This application uses a fine-tuned multilingual BERT model to calculate semantic similarity between sentences.</p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM fully loaded and parsed');
            document.getElementById('similarityForm').addEventListener('submit', async function(e) {
                console.log('Form submitted');
                e.preventDefault();
                
                const sentence1 = document.getElementById('sentence1').value;
                const sentence2 = document.getElementById('sentence2').value;
                
                console.log(`Processing sentences: '${sentence1}' and '${sentence2}'`);
                
                // Show loading spinner
                document.getElementById('loadingSpinner').style.display = 'block';
                
                try {
                    console.log('Sending fetch request to /similarity');
                    const response = await fetch('/similarity', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            sentence1: sentence1,
                            sentence2: sentence2
                        })
                    });
                    
                    console.log('Received response:', response);
                    const data = await response.json();
                    console.log('Parsed JSON data:', data);
                    
                    // Hide loading spinner
                    document.getElementById('loadingSpinner').style.display = 'none';
                    
                    // Show result
                    const resultContainer = document.getElementById('resultContainer');
                    resultContainer.style.display = 'block';
                    
                    // Get the raw score from the API (assuming it's from 0 to 5 now)
                    const score = data.similarity_score;
                    
                    // Update the score display
                    document.getElementById('scoreValue').textContent = score.toFixed(3);
                    document.getElementById('scoreDisplay').textContent = score.toFixed(3);
                    
                    // Calculate percentage for progress bar (from 0-5 to 0-100%)
                    const normalizedScore = Math.max(0, Math.min(5, score)) * 20; // Convert from 0-5 to 0-100%
                    
                    // Update progress bar
                    const progressBar = document.getElementById('scoreProgressBar');
                    progressBar.style.width = `${normalizedScore}%`;
                    progressBar.setAttribute('aria-valuenow', score);
                    
                    // Set color based on score
                    if (score < 1.67) {
                        progressBar.className = 'progress-bar bg-danger';
                    } else if (score < 3.33) {
                        progressBar.className = 'progress-bar bg-warning';
                    } else {
                        progressBar.className = 'progress-bar bg-success';
                    }
                    
                    // Add interpretation
                    let interpretation = '';
                    if (score < 1.67) {
                        interpretation = 'These sentences have low semantic similarity.';
                    } else if (score < 3.33) {
                        interpretation = 'These sentences have moderate semantic similarity.';
                    } else {
                        interpretation = 'These sentences have high semantic similarity.';
                    }
                    document.getElementById('interpretationText').textContent = interpretation;
                    
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('loadingSpinner').style.display = 'none';
                    alert('An error occurred while calculating similarity. Please try again.');
                }
            });
        });
    </script>
</body>
</html>""")
print(f"File HTML statico creato in {static_html_path}")

# Initialize tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    # Load the model with the trained weights
    model = BERTSimilarityModel()
    model.load_state_dict(torch.load("bert_similarity_model_weights.pth", map_location=torch.device("cpu")))
    model.eval()
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

def compute_similarity_score(sentence1, sentence2):
    inputs = tokenizer(
        sentence1, 
        sentence2, 
        padding='max_length', 
        truncation='longest_first', 
        max_length=32, 
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
    score = output.item()
    return round(score, 3)

@app.route("/")
def home():
    print("Home route accessed!")
    return send_from_directory(static_dir, "index.html")

@app.route("/similarity", methods=["POST"])
def similarity():
    print("Similarity endpoint accessed!")
    data = request.get_json()
    sentence1 = data["sentence1"]
    sentence2 = data["sentence2"]
    print(f"Processing sentences: '{sentence1}' and '{sentence2}'")
    # Compute the similarity score using the function
    score = compute_similarity_score(sentence1, sentence2)
    print(f"Similarity score: {score}")
    return jsonify({"similarity_score": score})

if __name__ == "__main__":
    app.run(debug=True)
