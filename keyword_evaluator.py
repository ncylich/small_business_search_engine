import webscraper as ws
import pandas as pd
import regex as re
import nltk
import json

REDUCE = False
if REDUCE:
    nltk.download('punkt')

# FOR All FUNCTIONS: the reduce option is to reduce the keywords to their root words before processing (optional)


# This function reads a json file and returns a dictionary of the correct type for the evaluate function
def json_to_dict(jsonpath: str, reduce=False) -> dict:
    with open(jsonpath, "r") as file:
        data = json.load(file)
    modded_data = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Value for key {key} is not a string")
        if not isinstance(value, int):
            raise ValueError(f"Key {key} is not an int")
        if reduce:
            key = nltk.word_tokenize(key.lower())
            key = ' '.join(key)
            modded_data[key] = int(value)
        else:
            modded_data[key.lower()] = int(value)
    return modded_data


# This function writes a dictionary to a json file
def dict_to_json(data: dict, jsonpath: str):
    with open(jsonpath, "w") as file:
        json.dump(data, file)


# This function takes a map of <word, int>
# then counts the num of times each word appears in the sentence and adds the int value to the score for each occurrence
def evaluate(keywords: dict, sentence: str, reduce=False) -> int:
    sentence = sentence.lower()
    if reduce:
        sentence = nltk.word_tokenize(sentence)
        sentence = ' '.join(sentence)

    score = 0
    for keyword, value in keywords.items():
        regex = r'\b' + keyword + r'\b'
        matches = re.findall(regex, sentence, re.IGNORECASE)
        score += value * len(matches)

        # adding bias valuing just the single match of the word
            # think like would you rather have 1 word matched 10 times, or 10 words matched once each
                # I think you'd rather have the 10 words mathced once, so this factor is to account for that
                    # so if a word occurs at least once, it has the weight of 3 occurances
        if len(matches) > 0:
            score += value * 2
    return score


def calculate_score(row, jsonpath="udu_keywords.json", reduce=False):
    # Load keywords
    keywords = json_to_dict(jsonpath, reduce=reduce)

    # Calculate score for the row
    url = row['Company']
    text = ws.search(url)
    score = 1e6 * evaluate(keywords, text, reduce=reduce) / len(text)
    print(f'Score for "{url}" is {score}')  # for dtesting purposes only

    return score


if __name__ == '__main__':
    # This is an example of how to use the functions
    keywords = json_to_dict("udu_keywords.json", reduce=REDUCE)
    udu = pd.read_csv("udu.csv")
    udu = udu.head(10)  # for testing purposes only
    udu['Score'] = udu.apply(calculate_score, axis=1)
    udu.to_csv("udu_with_custom_score.csv", index=False)
