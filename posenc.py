import numpy as np

def posenc(seq_len, ed):
    """
    Positional encoding for a given sequence length and embedding dimension.

    Args:
        seq_len (int): The length of the sequence.
        ed (int): The embedding dimension.

    Returns:
        np.ndarray: The positional encoding matrix of shape (seq_len, ed).
    """
    pos_enc = np.zeros((seq_len, ed))
    for pos in range(seq_len):
        for i in range(ed):
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / ed)))
            else:
                pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / ed)))
    return pos_enc

# Example usage
embed_dim = 100  # Embedding dimension
pe = posenc(1114, embed_dim)
print(pe)