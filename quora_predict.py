import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model_and_components(model_path, tokenizer_path, max_length_path):
    # Load the model
    model = load_model(model_path)
    
    # Load the tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    # Load max_length
    with open(max_length_path, 'r') as f:
        max_length = json.load(f)['max_length']
    
    return model, tokenizer, max_length

def preprocess_text(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences

def predict_spam(text, model, tokenizer, max_length):
    processed_text = preprocess_text(text, tokenizer, max_length)
    prediction = model.predict(processed_text)[0][0]
    return prediction

# Load model and components
model_path = 'spam_classifier_model.keras'
tokenizer_path = 'tokenizer.json'
max_length_path = 'max_length.json'
model, tokenizer, max_length = load_model_and_components(model_path, tokenizer_path, max_length_path)

sample_texts = [
    "Are pakistani's slave of chinese",
    "do you know any corrupt officiers in your areas",
    "Are you aware of any mentally retarded indians",
    "who will win in indian elections bjp or congress"
]

for text in sample_texts:
    spam_probability = predict_spam(text, model, tokenizer, max_length)
    print(f"Text: {text}")
    print(f"Spam Probability: {spam_probability:.4f}")
    print(f"Classification: {'Spam' if spam_probability > 0.5 else 'Not Spam'}")
    print()

# option for user to enter questions
while True:
    user_input = input("Enter a text to classify (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    spam_probability = predict_spam(user_input, model, tokenizer, max_length)
    print(f"Spam Probability: {spam_probability:.4f}")
    print(f"Classification: {'Spam' if spam_probability > 0.5 else 'Not Spam'}")
    print()
