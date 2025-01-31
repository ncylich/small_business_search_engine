import re
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler


# CKR Comparison Type
COMP_MULT = True  # True for multiplication, False for addition comparison methods

tqdm.pandas(desc="Processing Rows")

# Load tokenizer and model from Hugging Face Hub
model_name = "sentence-transformers/roberta-base-nli-stsb-mean-tokens"  # bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)


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


# This function writes a dictionary to a json file
def dict_to_json(data: dict, jsonpath: str):
    with open(jsonpath, "w") as file:
        json.dump(data, file)


# Generate BERT embeddings for a given text
def get_bert_embedding(text):
    # Encode text to get token ids and attention masks
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate output from the model
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Average the token embeddings to get a single vector for each input
    embedding = outputs.last_hidden_state.mean(dim=1)
    # Normalize the embedding
    norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
    normalized_embedding = embedding / norm
    return normalized_embedding


# Compare two embeddings using cosine similarity * Euclidean distance
def bert_compare_mult(key_phrase_embedding, sentence_embedding):
    # euclidean_distances(key_phrase_embedding, sentence_embedding).item()  # TODO: Incorporate Euclidean distance
    # return cosine_similarity(key_phrase_embedding, sentence_embedding).item()
    return (euclidean_distances(key_phrase_embedding, sentence_embedding).item() *
            cosine_similarity(key_phrase_embedding, sentence_embedding).item())


# Compare two embeddings using weighted cosine similarity + Euclidean distance
def bert_compare_add(key_phrase_embedding, sentence_embedding, cos_over_euc=0.5):
    c_sim = cosine_similarity(key_phrase_embedding, sentence_embedding).item() + 1  # normalized from 0 to 2
    e_dist = euclidean_distances(key_phrase_embedding, sentence_embedding).item()
    c_sim /= 2  # normalized from 0 to 1
    e_dist = 1 / (1 + e_dist)  # normalized from 0 to 1
    return (c_sim * cos_over_euc) + (e_dist * (1 - cos_over_euc))  # weighted average


def bert_compare(key_phrase_embedding, sentence_embedding):
    return bert_compare_mult(key_phrase_embedding, sentence_embedding) if COMP_MULT \
        else bert_compare_add(key_phrase_embedding, sentence_embedding)


def split_into_sentences(paragraph):
    # looks for periods, question marks, and exclamation marks followed by a space
    # and optionally followed by an uppercase letter (to handle abbreviations).
    pattern = r'(?<=[.?!])\s+(?=[A-Z])|(?<=[.?!])\s+(?=[a-z])'
    return re.split(pattern, paragraph)


def generate_dict_embeddings(data: dict):
    temp_dict = {}
    for key, value in data.items():
        temp_dict[key] = (get_bert_embedding(key), value)
    return temp_dict


def tf_idf_score(udu: pd.DataFrame, keywords: dict, score_col: str = 'Key Score', drop: bool = True):
    # calculating tf-idf for each row over all key phrases
    count = 1
    for key, (embedding, weight) in keywords.items():
        print(f"Calculating tf-idf for {key}: {count} of {len(keywords)})")

        def calculate_similarity(row):
            return bert_compare(embedding, row['Description Embedding'])

        udu[key] = udu.progress_apply(calculate_similarity, axis=1)
        idf = math.log(udu.shape[0] / udu[key].sum())
        udu[key] = udu[key] * idf * weight  # multiplying by idf and weight
        count += 1

    # calculating final tf-idf score
    def calculate_tfidf_score(row):
        return sum([row[key] for key in keywords.keys()])

    udu[score_col] = udu.progress_apply(calculate_tfidf_score, axis=1)
    udu[score_col] = MinMaxScaler().fit_transform(udu[[score_col]])  # linear 0-1 normalization

    if drop:
        udu.drop(columns=[keywords.keys()], inplace=True, axis=1)  # removing the keyword columns

    return udu


def evaluate(udu: pd.DataFrame, keywords: dict, sector: str = "", result_path="udu_scores.csv"):
    # Preprocessing / Initialization
    udu = udu.dropna(subset=['Description'])  # removing rows with empty descriptions
    keywords = generate_dict_embeddings(keywords)

    sectors = split_into_sentences(sector)
    sectors = {sect: 1 for sect in sectors}
    sector_keys = generate_dict_embeddings(sectors)

    # Calculating embedding for each description
    def calculate_description_embedding(row):
        return get_bert_embedding(row['Description'])

    udu['Description Embedding'] = udu.progress_apply(calculate_description_embedding, axis=1)

    # calculating tf-idf scores for sector's sentences and the key phrases
    udu = tf_idf_score(udu, sector_keys, 'Sector Score')
    udu = tf_idf_score(udu, keywords, 'Key Score')

    # calculating final score
    def calculate_score(row):
        return row['TF-IDF Score'] * row['Sector Score'] if sector else row['TF-IDF Score']

    udu['Score'] = udu.progress_apply(calculate_score, axis=1)
    udu['Score'] = MinMaxScaler().fit_transform(udu[['Score']])  # linear 0-1 normalization

    # udu.drop(columns=[keywords.keys()], inplace=True, axis=1)  # removing the keyword columns (optional)
    udu.drop(columns=['Description Embedding'], inplace=True, axis=1)  # dropping "Description Embedding" column

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
