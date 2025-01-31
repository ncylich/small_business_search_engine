# import comp_keyword_ranking as ckr
import batched_comparison as ckr
import pandas as pd
from tqdm import tqdm
import os

# TODO: Better email checking, improve AIRS & Make GPT Outputs more human
# TODO: For deployment, implement all params into argparse

USE_WLYB = False
if USE_WLYB:
    import gpt_description as gpt

tqdm.pandas(desc="Processing Rows")

# CKR Params
UDU_CSV = "udu_list.csv"
SECTOR = None
RESULTSPATH = "udu_top_results.csv"
JSONPATH = "udu_keywords.json"

# WLYB Params
INSTRUCTIONS = "instructions.txt"
OUTPUTS = "output"
MODEL = "gpt-4o"

CHECK_EMAIL = True

# choose_top_n Param
MIN_TOKENS = 20
N_COMPANIES = 500


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    repeat_cols = [col for col in cols if col.endswith('_y')]
    df = df.drop(repeat_cols, axis=1)
    rename_cols = {col for col in cols if col.endswith('_x')}
    rename_cols = {col: col[:-2] for col in rename_cols}
    df = df.rename(columns=rename_cols)
    return df


def get_pos_name_email(row: pd.Series, idx: int) -> list:
    pos = row[f'Apollo contact data_title_{idx}']
    fname = row[f'Apollo contact data_first_name_{idx}']
    lname = row[f'Apollo contact data_last_name_{idx}']
    email = row[f'Apollo contact data_email_{idx}']
    return [pos, fname, lname, email]


def format_person(person: list) -> list:
    pos, fname, lname, email = person
    if not pos or isinstance(pos, float):
        pos = ""
    if not fname or isinstance(fname, float):
        fname = ""
    if not lname or isinstance(lname, float):
        lname = ""
    if not email or isinstance(email, float):
        email = ""
    return [pos.strip().lower(), fname.strip().lower(), lname.strip().lower(), email.strip().lower()]


def check_email(person: list) -> list:
    pos, fname, lname, email = person
    email = email.split('@')
    if (len(email) != 2) or ('.' not in email[1]):
        return False
    email = email[0].lower().strip()
    if fname in email or lname in email:
        return True
    elif len(email) <= 3:
        if fname[0] in email and lname[0] in email:
            return True
    return False


def update_row(row: pd.Series, person: list, idx: int) -> None:
    pos, fname, lname, email = person
    row[f'Apollo contact data_title_{idx}'] = pos
    row[f'Apollo contact data_first_name_{idx}'] = fname
    row[f'Apollo contact data_last_name_{idx}'] = lname
    row[f'Apollo contact data_email_{idx}'] = email


def filter_contacts(row: pd.Series) -> pd.Series:
    people = [get_pos_name_email(row, i) for i in range(1, 4)]  # getting people
    people = [format_person(person) for person in people]  # formatting people
    people = [person for person in people if (person[1] or person[2]) and person[3]]  # filtering out empty people
    people = [person for person in people if check_email(person)]  # filtering out invalid emails
    def pos_sort(person):
        pos = person[0].strip().lower()
        if "owner" in pos or "ceo" in pos or "chief executive officer" in pos:
            return 0
        if "founder" in pos:
            return 1
        if "vp" in pos or "vice president" in pos or "vice-president" in pos:
            return 3
        if "president" in pos or "executive" in pos:
            return 2
        return 4

    people = sorted(people, key=pos_sort) # sorting people by position
    # updating the row
    for i, person in enumerate(people):
        update_row(row, person, i + 1)
    # filling in the rest of the blank rows
    for i in range(len(people), 3):
        update_row(row, ["", "", "", ""], i + 1)

    return row


def check_validity(row: pd.Series) -> bool:
    email = row['Apollo contact data_email_1']
    if isinstance(email, str) and email:
        return True
    return False


def choose_top_n(df: pd.DataFrame, min_tokens: int = 20, n_companies: int = -1) -> pd.DataFrame:
    # Remove rows where the number of tokens in 'description' is less than 'min_tokens'
    n_companies = min(100, int(.10 * len(df))) if n_companies < 0 else n_companies
    df['token_length'] = df['Description'].apply(lambda x: len(x.split()))
    df = df[df['token_length'] >= min_tokens]

    # Optionally, you can drop the 'token_length' column if it's no longer needed
    df = df.drop('token_length', axis=1)
    return df.head(n_companies)


def pipeline(udu_csv=UDU_CSV, results_path=RESULTSPATH, keys=JSONPATH, dir_path="", sector=SECTOR, instructions=INSTRUCTIONS,
             outputs=OUTPUTS, model=MODEL, verify_email=True, min_tokens=20, n_companies=500, rename=True):
    # Setting up the paths
    udu_csv = os.path.join(dir_path, udu_csv) if dir_path else udu_csv
    results_path = os.path.join(dir_path, results_path) if dir_path else results_path

    udu = pd.read_csv(udu_csv)  # reading the UDU data
    udu = rename_cols(udu) if rename else udu  # renaming columns as specified

    # verifying emails
    if verify_email:
        udu = udu.progress_apply(filter_contacts, axis=1)  # checking if the email is valid
        udu['valid_email'] = udu.apply(check_validity, axis=1)  # checking if the email is valid
        udu = udu[udu['valid_email'] == True]  # filtering out invalid emails
        udu = udu.drop('valid_email', axis=1)
    # udu = udu.head(128) if __name__ == "__main__" else udu  # for testing purposes
    print("Using", len(udu), "companies...")

    # getting ranked udu companies
    print("Evaluating UDU companies...")
    if isinstance(keys, str):
        keys = ckr.json_to_dict(keys)
    udu = ckr.evaluate(udu, keys, sector, results_path)  # still ~20k companies
    udu = choose_top_n(udu, min_tokens, n_companies)  # now ~500 companies
    udu.to_csv(results_path, index=False)  # saving the final output

    # generating descriptions
    if USE_WLYB:
        print("Generating WLYBs...")
        config = gpt.get_model_config(instructions, outputs, model)
        udu = gpt.generate_wlyb_column(udu, config)  # generating the WLYBs
        udu['WLYB'] = udu['WLYB'].str.extract('(of .*)', expand=False)

    # Final Results
    udu.to_csv(results_path, index=False)  # saving the final output


if __name__ == "__main__":
    pipeline("udu.csv", "ranked_udu.csv", verify_email=False)
