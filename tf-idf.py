import math


# Assume 'documents' is a list of tuples (f, d_f) where f is the firm identifier and d_f is the document (description of the firm)
# Assume 'keywords' is a list of keywords to analyze
# Assume 'weights' is a dictionary mapping keywords to their respective weights


def term_frequency(term, doc):
    return doc.count(term) / len(doc.split())


def inverse_document_frequency(term, docs):
    num_docs_containing_term = sum(1 for f, d_f in docs if term in d_f)
    return math.log(len(docs) / (1 + num_docs_containing_term))


def compute_tf_idf(documents, keywords):
    tf_idf_scores = {f: {key: 0 for key in keywords} for f, _ in documents}
    idf_scores = {key: inverse_document_frequency(key, [d_f for _, d_f in documents]) for key in keywords}

    for f, d_f in documents:
        for key in keywords:
            tf = term_frequency(key, d_f)
            tf_idf_scores[f][key] = tf * idf_scores[key]

    return tf_idf_scores


def calculate_scores(documents, keywords):
    tf_idf_scores = compute_tf_idf(documents, keywords)
    scores = {f: sum(weights[key] * tf_idf_scores[f][key] for key in keywords) for f, _ in documents}
    min_score, max_score = min(scores.values()), max(scores.values())
    normalized_scores = {f: (score - min_score) / (max_score - min_score) for f, score in scores.items()}
    return normalized_scores


# Example usage:
documents = [('firm1', 'document text here'), ('firm2', 'another document text')]
keywords = ['keyword1', 'keyword2']
weights = {'keyword1': 1.5, 'keyword2': 2.0}

# Normalized scores for each firm
normalized_scores = calculate_scores(documents, keywords, weights)
print(normalized_scores)