import re
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler

# CKR Comparison Type
COMP_MULT = True  # True for multiplication, False for addition comparison methods
TEST_MODE = False

tqdm.pandas(desc="Processing Rows")

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


def json_to_dict(jsonpath: str) -> dict:
    with open(jsonpath, "r") as file:
        data = json.load(file)
    modded_data = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Value for key {key} is not a string")
        if not isinstance(value, int):
            raise ValueError(f"Key {key} is not an int")
        modded_data[key.lower()] = int(value)
    return modded_data


def dict_to_json(data: dict, jsonpath: str):
    with open(jsonpath, "w") as file:
        json.dump(data, file)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
    normalized_embedding = embedding / norm
    return normalized_embedding


def bert_compare_mult(key_phrase_embeddings, sentence_embeddings):
    euclidean_dist = torch.cdist(key_phrase_embeddings, sentence_embeddings)
    cosine_sim = torch.nn.functional.cosine_similarity(key_phrase_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=2)
    return euclidean_dist * (1 + cosine_sim) / 2


def bert_compare_add(key_phrase_embeddings, sentence_embeddings, cos_over_euc=0.5):
    cosine_sim = torch.nn.functional.cosine_similarity(key_phrase_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=2)
    euclidean_dist = torch.cdist(key_phrase_embeddings, sentence_embeddings)
    cosine_sim = (cosine_sim + 1) / 2  # Normalize to [0, 1]
    euclidean_dist = 1 / (1 + euclidean_dist)  # Normalize to [0, 1]
    return cos_over_euc * cosine_sim + (1 - cos_over_euc) * euclidean_dist


def bert_compare(key_phrase_embeddings, sentence_embeddings):
    if COMP_MULT:
        return bert_compare_mult(key_phrase_embeddings, sentence_embeddings)
    else:
        return bert_compare_add(key_phrase_embeddings, sentence_embeddings)


def split_into_sentences(paragraph):
    pattern = r'(?<=[.?!])\s+(?=[A-Z])|(?<=[.?!])\s+(?=[a-z])'
    return re.split(pattern, paragraph)


def generate_dict_embeddings(data: dict):
    temp_dict = {}
    for key, value in data.items():
        temp_dict[key] = (get_bert_embedding(key), value)
    return temp_dict


def batch_process(data, batch_size, process_function):
    num_batches = (len(data) + batch_size - 1) // batch_size if not TEST_MODE else 1
    for i in tqdm(range(num_batches)):
        yield process_function(data[i*batch_size : (i+1)*batch_size])
    if TEST_MODE:
        empty_data = pd.DataFrame({"Description": [""] * len(data)})
        for i in range(num_batches, (len(data) + batch_size - 1) // batch_size):
            yield process_function(empty_data[i*batch_size : (i+1)*batch_size])


def tf_idf_vectorized(descs: torch.Tensor, keywords: dict):
    key_embeddings = torch.cat([embedding for embedding, _ in keywords.values()]).to(device)
    weights = torch.tensor([weight for _, weight in keywords.values()]).to(device)

    # Vectorized similarity calculation
    similarities = bert_compare(key_embeddings, descs)
    # print(similarities.shape, descs.shape)
    idfs = torch.log(descs.shape[0] / similarities.sum(dim=1, keepdim=True))
    # print(similarities.shape, idfs.shape, weights.shape, weights)
    tf_idf_scores = similarities * idfs * weights.unsqueeze(1)
    # print(tf_idf_scores.shape, tf_idf_scores)
    final_scores = tf_idf_scores.sum(dim=0)

    return final_scores


def evaluate(udu: pd.DataFrame, keywords: dict, sector: str = "", result_path="udu_scores.csv"):
    udu = udu.dropna(subset=['Description'])
    keywords = generate_dict_embeddings(keywords)

    if sector:
        sectors = split_into_sentences(sector)
        sectors = {sect: 1 for sect in sectors}
        sector_keys = generate_dict_embeddings(sectors)

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
    all_embeddings = torch.cat(all_embeddings).to(device)

    udu = udu.copy()  # gets rid of copy error by removng aliasing
    keyword_similarity = tf_idf_vectorized(all_embeddings, keywords)
    udu['Keyword Score'] = keyword_similarity.cpu().numpy()

    if sector:
        sector_similarity = tf_idf_vectorized(all_embeddings, sector_keys)
        udu['Sector Score'] = sector_similarity.cpu().numpy()

    udu['Score'] = udu['Keyword Score'] + udu['Sector Score'] if sector else udu['Keyword Score']

    udu['Score'] = MinMaxScaler().fit_transform(udu[['Score']])
    udu = udu.sort_values(by='Score', ascending=False)
    if result_path is not None:
        udu.to_csv(result_path, index=False)
    return udu


def old_evaluate(udu: pd.DataFrame, sector=None, result_path="udu_scores.csv", jsonpath="udu_keywords.json"):
    return evaluate(udu, json_to_dict(jsonpath), sector, result_path)


if __name__ == '__main__':
    df = pd.read_csv('udu.csv')
    keys = json_to_dict('udu_keywords.json')
    evaluate(df, keys, "", 'udu_top_scores.csv')
