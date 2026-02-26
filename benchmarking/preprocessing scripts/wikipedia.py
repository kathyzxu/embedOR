"""
Preprocess and normalize Wikipedia Simple English dataset using Cohere.ai embedding model. save to .npy.
"""

from datasets import load_dataset
import numpy as np
import os

repo_root = os.path.dirname(os.path.dirname(__file__)) 
output_folder = os.path.join(repo_root, "preprocessed_data")
os.makedirs(output_folder, exist_ok=True)

docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

embeddings = []

for doc in docs:
    emb = np.array(doc['emb'], dtype=np.float32)
    emb /= np.linalg.norm(emb)  # L2 normalization
    embeddings.append(emb)

embeddings = np.stack(embeddings)
np.save(os.path.join(output_folder, "wikipedia.npy"), embeddings)
print("Saved normalized embeddings to:", output_folder)
