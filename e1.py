import cupy as np
from collections import defaultdict
import math

# Sample corpus
corpus = [
    "i like deep learning",
    "i like NLP",
    "i enjoy flying"
]

# Parameters
window_size = 2
embedding_size = 10
alpha = 0.75
x_max = 100
learning_rate = 0.05
epochs = 100

# Build vocabulary
word2id = {}
id2word = {}
for sentence in corpus:
    for word in sentence.split():
        if word not in word2id:
            idx = len(word2id)
            word2id[word] = idx
            id2word[idx] = word
vocab_size = len(word2id)

# Build co-occurrence matrix
co_matrix = defaultdict(float)
for sentence in corpus:
    words = sentence.split()
    for i, word in enumerate(words):
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(words))):
            if i != j:
                wi, wj = word2id[word], word2id[words[j]]
                co_matrix[(wi, wj)] += 1.0 / abs(i - j)

# Initialize embeddings and biases
W = np.random.randn(vocab_size, embedding_size) * 0.01
W_context = np.random.randn(vocab_size, embedding_size) * 0.01
biases = np.zeros(vocab_size)
biases_context = np.zeros(vocab_size)
print(co_matrix)
# Training loop
for epoch in range(epochs):
    total_loss = 0
    for (i, j), Xij in co_matrix.items():
        # Weighting function
        f_x = (Xij / x_max) ** alpha if Xij < x_max else 1

        # Dot product + biases
        dot = np.dot(W[i], W_context[j]) + biases[i] + biases_context[j]
        loss = f_x * ((dot - np.log(Xij)) ** 2)

        # Gradients
        grad = 2 * f_x * (dot - np.log(Xij))
        W[i] -= learning_rate * grad * W_context[j]
        W_context[j] -= learning_rate * grad * W[i]
        biases[i] -= learning_rate * grad
        biases_context[j] -= learning_rate * grad

        total_loss += 0.5 * loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Combine word and context vectors
embeddings = W + W_context

# Example: cosine similarity between words
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

word1, word2 = "like", "enjoy"
sim = cosine_similarity(embeddings[word2id[word1]], embeddings[word2id[word2]])
print(f"Similarity between '{word1}' and '{word2}': {sim:.4f}")
