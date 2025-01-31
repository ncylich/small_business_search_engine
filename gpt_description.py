import argparse
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import os

client = OpenAI()

def few_shot_model(prompt, model_config):
    context = [{"role": "system", "content": model_config["instructions"]}]
    for example in zip(model_config["example_input"], model_config["example_output"]):
        single_shot = [{"role": "user", "content": example[0]},
                       {"role": "assistant", "content": example[1]}]
        context += single_shot
    context.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model_config["model_name"],
        messages=context
    )
    return completion.choices[0].message.content

def format_row_for_gpt(row):
    formatted_input = ", ".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
    return formatted_input

def get_model_config(instructions, output_paradigm, model_name="gpt-4-0125-preview"):
    fn = os.path.join('gpt_info', instructions)
    with open(fn, "r") as f:
        instructions = f.read()

    example_input, example_output = [], []
    i = 0
    while True:
        fn = os.path.join('gpt_info', f"input{i + 1}.txt")
        if not os.path.exists(fn):
            break
        with open(fn, "r") as f:
            example_input.append(f.read())
        fn = os.path.join('gpt_info', output_paradigm + f"{i + 1}.txt")
        with open(fn, "r") as f:
            example_output.append(f.read())
        i += 1

    if i == 0:
        raise Exception("Error: No input files found for few-shot model")

    model_config = {"instructions": instructions,
                    "example_input": example_input,
                    "example_output": example_output,
                    "model_name": model_name}
    return model_config

def add_wlyb_column(row, model_config):
    formatted_input = format_row_for_gpt(row)
    gpt_output = few_shot_model(formatted_input, model_config)
    return gpt_output

def generate_wlyb_column(df, model_config, rows=None):
    if rows:
        df = df.head(rows)
    tqdm.pandas(desc=f"WLYB for {len(df)} rows")
    df['WLYB'] = df.progress_apply(add_wlyb_column, axis=1, args=(model_config,))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate WLYB column based on GPT-4 model")
    # parser.add_argument("--model", type=int, choices=[1, 2], required=True, help="Choose model 1 or 2")
    parser.add_argument("--rows", type=int, help="Number of rows to process (default is all rows)")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file")

    args = parser.parse_args()

    model_config_short = get_model_config("instructions.txt",
                                          "output",
                                          model_name="gpt-4o")

    udu = pd.read_csv(args.csv)

    # Determine the subset of rows to process
    if args.rows:
        subset = udu.iloc[:args.rows]
        subset_indexes = subset.index
    else:
        subset = udu
        subset_indexes = udu.index

    # Apply the function to the subset of rows
    tqdm.pandas(desc=f"Processing {len(subset)} rows")
    udu.loc[subset_indexes, 'WLYB'] = subset.progress_apply(add_wlyb_column, axis=1, args=(model_config_short,))
    # udu.loc[subset_indexes, 'WLYB Long'] = subset.progress_apply(add_wlyb_column, axis=1, args=(model_config_long,))

    udu.to_csv(f"{args.csv[:-4]}_wlyb.csv", index=False)
