import batched_comparison as ckr
import torch
import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc="Processing Rows")

# TODO: tqdm progress bar

def process_embeddings(text):
    inputs = ckr.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(ckr.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = ckr.bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings


def calculate_description_embedding(batch):
    descriptions = batch['Description'].tolist()
    return process_embeddings(descriptions)


def calculate_sector_embeddings(batch):
    sectors = batch['Sector'].tolist()
    return process_embeddings(sectors)


def calculate_embeddings(df: pd.DataFrame, tp: str = "description", batch_size: int = 64):
    embeddings = []
    func = calculate_sector_embeddings if tp.strip().lower() == "sector" else calculate_description_embedding
    for batch in ckr.batch_process(df, batch_size, func):
        embeddings.append(batch)
    return torch.cat(embeddings).to(ckr.device)


def main():
    sectors = pd.read_csv("sector_list.csv")
    udu = pd.read_csv("udu_list.csv")

    description_embeddings = calculate_embeddings(udu, "description")
    sector_embeddings = calculate_embeddings(sectors, "sector")

    # Must compare all companies to all sectors
    similarities = ckr.bert_compare(description_embeddings, sector_embeddings)

    # chose the best sector for each company
    sector_scores = similarities.max(dim=1).values.cpu().numpy()

    def get_sector_name(row):
        idx = row.idxmax()
        return sectors.iloc[idx]['Sector']

    predicted_sectors = [get_sector_name(row) for row in sector_scores]
    udu['Predicted Sector'] = predicted_sectors

    # save results
    udu.to_csv("udu_predicted_sectors.csv", index=False)

    return udu


if __name__ == "__main__":
    main()
