import os
import runpod
from typing import Dict, List, Union
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Define constants
MODEL_DIR = "/app/model"
MAX_LENGTH = 128

# Define category mapping
CATEGORIES = {
    0: "Utilities",
    1: "Health",
    2: "Dining",
    3: "Travel",
    4: "Education",
    5: "Subscription",
    6: "Family",
    7: "Food",
    8: "Festivals",
    9: "Culture",
    10: "Apparel",
    11: "Transportation",
    12: "Investment",
    13: "Shopping",
    14: "Groceries",
    15: "Documents",
    16: "Grooming",
    17: "Entertainment",
    18: "Social Life",
    19: "Beauty",
    20: "Rent",
    21: "Money transfer",
    22: "Salary",
    23: "Tourism",
    24: "Household",
}

# Load model and tokenizer at container startup
def load_model():
    print("Loading BERT transaction categorization model...")
    try:
        # First try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        
        # Try loading the model with specific parameters
        model = BertForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            low_cpu_mem_usage=True  # Helps with memory usage during loading
        )
        
        print("Model loaded successfully!")
        return model, tokenizer, CATEGORIES
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Try with device_map to auto-manage memory
        try:
            print("Attempting alternative loading method...")
            model = BertForSequenceClassification.from_pretrained(
                MODEL_DIR,
                local_files_only=True,
                device_map="auto"  # Automatically manages memory
            )
            print("Model loaded successfully with alternative method!")
            return model, tokenizer, CATEGORIES
            
        except Exception as e2:
            print(f"Fatal error loading model: {e2}")
            raise e2

# Initialize model
model, tokenizer, id2label = load_model()

def predict_category(text, top_k=1):
    """
    Predict transaction category for the given text.
    
    Args:
        text (str): Transaction description text
        top_k (int): Number of top predictions to return
        
    Returns:
        list: Top k predictions with labels and scores
    """
    # Format input as expected by the model
    # The model was trained on transaction descriptions with a specific format
    if not text.lower().startswith("transaction:"):
        text = f"Transaction: {text} - Type: expense"
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get top-k predictions
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[1]))
    
    # Format results
    predictions = []
    for i, (score, idx) in enumerate(zip(values[0].tolist(), indices[0].tolist())):
        category = CATEGORIES.get(idx, f"Category_{idx}")
        predictions.append({
            "rank": i + 1,
            "category": category,
            "score": score
        })
    
    return predictions

def predict_batch(texts: List[str], top_k=1):
    """
    Predict categories for multiple transaction texts in a batch.
    
    Args:
        texts (List[str]): List of transaction description texts
        top_k (int): Number of top predictions to return for each transaction
        
    Returns:
        List[Dict]: List of prediction results for each transaction
    """
    results = []
    
    # Process each transaction text
    for text in texts:
        predictions = predict_category(text, top_k=top_k)
        results.append({
            "text": text,
            "predictions": predictions,
            "top_category": predictions[0]["category"] if predictions else None
        })
    
    return results

def handler(job: Dict) -> Dict:
    """
    RunPod handler function for processing transaction categorization requests.
    
    Args:
        job (dict): Job input containing either a single transaction text or a list of texts
        
    Returns:
        dict: Prediction results
    """
    job_input = job.get("input", {})
    
    # Get optional parameters
    top_k = job_input.get("top_k", 1)
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 1
    
    # Check if input is a batch of transactions or a single transaction
    if "texts" in job_input and isinstance(job_input["texts"], list):
        # Batch processing
        texts = job_input["texts"]
        if not texts or not all(isinstance(text, str) for text in texts):
            return {"error": "Invalid transaction texts provided"}
        
        try:
            batch_results = predict_batch(texts, top_k=top_k)
            return {"results": batch_results}
        except Exception as e:
            return {"error": f"Batch processing error: {str(e)}"}
    
    # Single transaction processing (for backward compatibility)
    elif "text" in job_input:
        text = job_input.get("text")
        if not text or not isinstance(text, str):
            return {"error": "No valid transaction text provided"}
        
        try:
            predictions = predict_category(text, top_k=top_k)
            return {
                "predictions": predictions,
                "top_category": predictions[0]["category"] if predictions else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    else:
        return {"error": "Input must contain either 'text' (string) or 'texts' (list of strings)"}

# Start the serverless worker
runpod.serverless.start({"handler": handler})