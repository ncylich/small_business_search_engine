import torch
import pandas as pd
from tqdm import tqdm
import keyword_evaluator as ke
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


#TODO: PRECALCULATE EMBEDDING WHEN LOADING KEYWORDS and ONLY GENERATE KEYWORD EMBEDDINGS DICTIONARY ONCE - STATIC GLOBAL
tqdm.pandas(desc="Processing Rows")


# Load tokenizer and model from Hugging Face Hub
'''
model_name = "sentence-transformers/roberta-base-nli-stsb-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
'''

# Bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

'''
# Load a pre-trained Sentence Transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')


def encode(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return embeddings


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
'''

def get_bert_embedding(text):
    # Encode text to get token ids and attention masks
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate output from the model
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Average the token embeddings to get a single vector for each input
    return outputs.last_hidden_state.mean(dim=1)


def bert_compare(keyphrase, sentence):

    # Compute embeddings
    keyphrase_embedding = get_bert_embedding(keyphrase)
    sentence_embedding = get_bert_embedding(sentence)

    # Compute cosine similarity
    cos_sim = cosine_similarity(keyphrase_embedding, sentence_embedding)
    # might want to add Euclidean similarity as well
    return cos_sim


def calculate_score(row, jsonpath="udu_keywords.json", reduce=False):
    keywords = ke.json_to_dict(jsonpath, reduce=reduce)

    # Calculate score for the row
    text = row['Description']
    score = 0
    for key, value in keywords.items():
        res = bert_compare(key, text).item()
        score += value * res

    #print(f'Score for {row["Company"]} is {score}')  # for testing purposes only
    return score

'''
def st_compare(keyphrase, sentence):
    # Generate embeddings
    keyphrase_embedding = st_model.encode(keyphrase)
    sentence_embedding = st_model.encode(sentence)

    # Calculate cosine similarity
    cos_sim = cosine_similarity([keyphrase_embedding], [sentence_embedding])[0][0]
    return cos_sim
'''

if __name__ == '__main__':
    '''
    REGENERATE = False
    if REGENERATE:
        udu = pd.read_csv("udu.csv")

        udu_starred = udu[udu['Starred'] == True]
        udu_not_starred = udu[udu['Starred'] == False]

        udu_starred = udu_starred.head(100)
        udu_not_starred = udu_not_starred.head(100)

        udu_starred['Score'] = udu_starred.progress_apply(calculate_score, axis=1)
        udu_not_starred['Score'] = udu_not_starred.progress_apply(calculate_score, axis=1)
    else:
        udu_starred = pd.read_csv("udu_starred_with_custom_score.csv")
        udu_not_starred = pd.read_csv("udu_not_starred_with_custom_score.csv")

    print(f"Average score for non-starred companies: {udu_not_starred['Score'].mean()} over {len(udu_not_starred)} companies with SD {udu_not_starred['Score'].std()}")
    print(f"Udu's avg score was {udu_not_starred['udu score'].mean()}")
    print(f"Average score for starred companies: {udu_starred['Score'].mean()} over {len(udu_starred)} companies with SD {udu_starred['Score'].std()}")
    print(f"Udu's avg score was {udu_starred['udu score'].mean()}")

    #udu.to_csv("udu_with_custom_score.csv", index=False)
    udu_starred.to_csv("udu_starred_with_custom_score.csv", index=False)
    udu_not_starred.to_csv("udu_not_starred_with_custom_score.csv", index=False)
    '''
    udu = pd.read_csv("udu.csv")
    udu['Score'] = udu.progress_apply(calculate_score, axis=1)
    udu = udu.sort_values(by='Score', ascending=False)
    udu = udu.head(500)
    udu.to_csv("udu_top_500_score.csv", index=False)
