# Install the datasets library (run this in your terminal or notebook)
# pip install datasets

from datasets import load_dataset

# Download and load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(dataset["text"])