import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    data['cleaned_text'] = data['question_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return data

# Plot histogram of sequence lengths
def plot_sequence_length_histogram(lengths):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Histogram of Sequence Lengths')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.axvline(x=np.median(lengths), color='r', linestyle='dashed', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
    plt.axvline(x=np.percentile(lengths, 95), color='g', linestyle='dashed', linewidth=2, label=f'95th Percentile: {np.percentile(lengths, 95):.0f}')
    plt.legend()
    plt.show()

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Create embedding matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Build model (balanced version)
def build_model(vocab_size, embedding_dim, embedding_matrix, max_length):
    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    # Load and preprocess data
    data = load_and_preprocess_data('train.csv')
    
    # Split data into train+val and test sets
    train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Further split train+val into train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=0.11111, random_state=42)
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['cleaned_text'])
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_data['cleaned_text'])
    val_sequences = tokenizer.texts_to_sequences(val_data['cleaned_text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['cleaned_text'])
    
    # Calculate sequence lengths
    train_lengths = [len(seq) for seq in train_sequences]
    
    # Plot histogram of sequence lengths
    plot_sequence_length_histogram(train_lengths)
    
    # Print some statistics
    print(f"Median sequence length: {np.median(train_lengths):.0f}")
    print(f"95th percentile of sequence length: {np.percentile(train_lengths, 95):.0f}")
    print(f"Max sequence length: {max(train_lengths)}")
    
    
    # Pad sequences
    train_padded = pad_sequences(train_sequences, maxlen=100)
    val_padded = pad_sequences(val_sequences, maxlen=100)
    test_padded = pad_sequences(test_sequences, maxlen=100)
    
    # Explicitly separate features and targets only for test data
    X_test, y_test = test_padded, test_data['target']
    
    # Load GloVe embeddings
    glove_file = 'glove.6B.100d.txt'  # Update with your GloVe file path
    embeddings_index = load_glove_embeddings(glove_file)
    
    # Create embedding matrix
    embedding_dim = 100  
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index, embedding_dim)
    
    # Build and train model
    model = build_model(len(tokenizer.word_index), embedding_dim, embedding_matrix, 100)
    history = model.fit(train_padded, train_data['target'], 
                        validation_data=(val_padded, val_data['target']),
                        epochs=10, batch_size=32)
    
    # Plot training history
   # plt.figure(figsize=(12, 6))
   # plt.plot(history.history['accuracy'], label='Training Accuracy')
   # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #plt.title('Model Accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()
    
    # Evaluate model on test set
    print("\nEvaluating model on test set:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {test_accuracy:.4f}')
    
    # Make predictions on test set
    test_predictions = model.predict(X_test)
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    
    # Calculate additional metrics
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions_binary))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions_binary))
    
    # Save predictions along with true labels
    test_results = pd.DataFrame({
        'true_label': y_test,
        'predicted_probability': test_predictions.flatten(),
        'predicted_label': test_predictions_binary.flatten()
    })
    test_results.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to test_predictions.csv")
    
    # Save the model
    model_save_path = 'spam_classifier_model.keras'  # Added .keras extension
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the tokenizer using JSON
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f)
    print("Tokenizer saved to tokenizer.json")

    # Save max_length
    with open('max_length.json', 'w') as f:
        json.dump({'max_length': 100}, f)
    print("max_length saved to max_length.json")

if __name__ == '__main__':
    main()