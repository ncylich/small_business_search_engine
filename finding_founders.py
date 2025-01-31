from openai import OpenAI
from copy import deepcopy

YOUR_API_KEY = "INSERT API KEY HERE"

system_message = {
    "role": "system",
    "content": (
        "You are an artificial intelligence assistant and you need to help the user find the founders of a company " +
        "they provide by searching the web. Answer in the followng form\nFounder 1: [Name], [Role]\n" +
        "Founder 2: [Name], [Role]\nFounder 3: [Name], [Role]\netc."
    ),
}

base_message = {
    "role": "user",
    "content": None,
}

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")


def request_owner_perplexity(company_name: str, model: str = "mistral-7b-instruct"):
    base_msg = deepcopy(base_message)
    base_msg["content"] = company_name

    # chat completion without streaming
    response = client.chat.completions.create(
        model=model,
        messages=[system_message, base_message],
    )
    print(response)
    return response


def parse_perp_response(response):
    response = response.choices[0].message.content # extract the response from the completion
    lines = response.split("\n") # split by newlines
    lines = [line.strip() for line in lines if line]  # remove empty lines

    # extract founders lines
    founders = [line for line in lines if line.startswith("Founder")]  # filter out non-founder lines
    founders = [founder.split(": ")[1] for founder in founders]  # extract founder names and roles
    founders = [founder.split(", ") for founder in founders]  # split names and roles
    founders = [founder for founder in founders if len(founder) > 1]  # filter out invalid founders

    # getting credientials
    names = [founder[0] for founder in founders] # extract names
    roles = [", ".join(founder[1:]) for founder in founders] # extract roles
    return zip(names, roles)


def get_founders(company_name: str, model: str = "mistral-7b-instruct"):
    response = request_owner_perplexity(company_name, model)
    return parse_perp_response(response)


'''
# chat completion with streaming
response_stream = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=messages,
    stream=True,
)
for response in response_stream:
    print(response)
'''