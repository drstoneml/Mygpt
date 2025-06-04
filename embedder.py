import math as m
import numpy as np

# Read the text file
with open("wikidata.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Preprocess the text
text = text.replace("\n", " ")

# Tokenize the text
def tokenize(text):
    return text.split()

tokens = tokenize(text)
vocab = sorted(set(tokens))  # Unique words in the vocabulary
vocab_size = len(vocab)  # Size of the vocabulary
embedding_dim = 30
print(f"Vocabulary Size: {vocab_size}")
# Create word-to-index and index-to-word mappings
w_to_idx = {word: idx for idx, word in enumerate(vocab)}  # Use vocab with unique words
idx_to_w = {idx: word for word, idx in w_to_idx.items()}

# One-hot encoding function
def one_hot_encode(word, word_to_index):
    vector = [0] * len(word_to_index)
    index = word_to_index[word]
    vector[index] = 1
    return vector

# Generate one-hot vectors for all tokens
one_hot_vectors = [one_hot_encode(word, w_to_idx) for word in tokens]
window_size = 2
training_data = []

for i, word in enumerate(tokens):
    target = one_hot_encode(word, w_to_idx)
    context_indices = list(range(max(0, i - window_size), i)) + \
                      list(range(i + 1, min(len(tokens), i + window_size + 1)))
    for j in context_indices:
        context = one_hot_encode(tokens[j], w_to_idx)
        training_data.append((target, context))
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)  # Apply softmax
def train(training_data, epochs, learning_rate, momentum=0.9):
    embedding_dim = 30
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)

    # Initialize velocity terms for momentum
    v_W1 = np.zeros_like(W1)
    v_W2 = np.zeros_like(W2)

    for epoch in range(epochs):
        loss = 0
        for target, context in training_data:
            # Forward pass
            h = np.dot(target, W1)
            u = np.dot(h, W2)
            y_pred = softmax(u)

            # Calculate error
            e = y_pred - context

            # Backpropagation
            dW2 = np.outer(h, e)  # Gradient for W2
            dW1 = np.outer(target, np.dot(W2, e))  # Gradient for W1

            # Update velocities (momentum)
            v_W2 = momentum * v_W2 - learning_rate * dW2
            v_W1 = momentum * v_W1 - learning_rate * dW1

            # Update weights using velocities
            W2 += v_W2
            W1 += v_W1

            # Compute loss (cross-entropy)
            loss += -np.sum(context * np.log(y_pred + 1e-9))

        print(f'Epoch {epoch+1}, Loss: {loss}')
    return W1, W2
W1, W2 = train(training_data, epochs=21, learning_rate=0.01)
def get_embedding(word):
    return W1[w_to_idx[word]]
np.save("W1.npy", W1)
