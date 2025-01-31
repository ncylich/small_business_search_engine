import os.path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from batched_comparison import batch_process

# Load tokenizer and model from Hugging Face Hub
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# Move the model to the GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    bert_model.to(device)
    print("Model loaded onto the MPS device.")
else:
    device = torch.device("cpu")
    print("MPS device not available, using CPU.")

def get_embeddings(udu: pd.DataFrame):
    udu = udu.dropna(subset=['Description'])
    def calculate_description_embedding(batch):
        descriptions = batch['Description'].tolist()
        inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings

    batch_size = 64  # Set batch size as needed
    all_embeddings = []
    for batch in batch_process(udu, batch_size, calculate_description_embedding):
        all_embeddings.append(batch)
    all_embeddings = torch.cat(all_embeddings)

    return all_embeddings.to('cpu').numpy()

if __name__ == "__main__":
    udu = pd.read_csv("udu.csv")
    print("getting embeddings")

    if os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy")
    else:
        embeddings = get_embeddings(udu)
        with open("embeddings.npy", "wb") as f:
            np.save(f, embeddings)

    print("clustering")
    arr = []
    for i in range(2, 100):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        print(i, score
        arr.append(score)
    # udu['Cluster'] = kmeans.labels_
    # udu.to_csv("udu_clusters.csv", index=False)
    # print("Clusters saved to udu_clusters.csv")