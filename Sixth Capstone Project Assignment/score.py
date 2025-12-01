import os
import json
import torch
import re
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def init():
    global model, tokenizer, device
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "my_final_bert_model")
    logging.info(f"Loading model from: {model_path}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise e

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        # 1. Get English text for Prediction
        input_list = data["inputs"] 
        if isinstance(input_list, str): input_list = [input_list]

        # 2. Get Original text for Display (Optional)
        # If 'original_text' is sent, use it. Otherwise, fallback to English input.
        display_list = data.get("original_text", input_list)

        # 3. Predict on ENGLISH text
        clean_texts = [clean_text(t) for t in input_list]
        inputs = tokenizer(clean_texts, padding=True, truncation=True, max_length=200, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        
        # 4. Return ORIGINAL text in the result
        results = []
        # Zip matches the first original text with the first prediction, and so on.
        for text, pred in zip(display_list, predictions):
            results.append({
                "text": text,       # This will now be Burmese
                "prediction": pred
            })
        
        return results
        
    except Exception as e:
        return {"error": str(e)}