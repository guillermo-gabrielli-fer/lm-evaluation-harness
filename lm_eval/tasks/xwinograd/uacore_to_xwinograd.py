"""
{
    "text":["Міські","чиновники","не","видали","протестувальникам","дозвіл",",","бо","вони","боялися","насилля","."],
    "options":[["міські","чиновники"],["протестувальники"]],
    "label":0,
    "pronoun":"вони",
    "pronoun_loc":8
}


Input:

ua-core
text - the Winograd schema in Ukrainian, tokenized
options - the two entity options that the pronoun may be referring to
label - the index of the correct option in options
pronoun - the pronoun in the sequence to be resolved
pronoun_loc - the index of the ambiguous pronoun in text


Destination:
sentence option1 option2 answer
"sentence": The city councilmen refused the demonstrators a permit because _ feared violence.
"option1":the demonstrators
"option2":The city councilmen
"answer": 2
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import re
import string
from huggingface_hub import login
from datasets import Dataset,DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert ua-coreref dataset to XWinograd format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=Path, default="../ua-coref/wsc-ua/wsc-ua.jsonlines", help='Input JSONL file path.')
    parser.add_argument('-o', '--output', type=Path, help='JSONL output')
    return parser.parse_args()

def convert_to_xwinograd(input_file: Path) -> pd.DataFrame:
    """Convert ua-coreref JSONL format to XWinograd format.

    Args:
        input_file (Path): Path to the input JSONL file.

    Returns:
        pd.DataFrame: DataFrame containing the converted data in XWinograd format.
    """
    data = []
    
    # remove extra spaces around punctuation
    p = re.compile(" ([" + re.escape(",.") + "])")
    print(p)

    with input_file.open('r') as f:
        for line in tqdm(f, desc='Processing lines'):
            entry = json.loads(line)
            # replace pronount to disambiguate with _
            entry['text'][entry['pronoun_loc']] = "_"
            text = ' '.join(entry['text'])
            # convert tokenized text back to plain text
            sentence = p.sub( r"\1",text).strip()
            options = entry['options']
            option1 = options[0][0]
            option2 = options[1][0]

            # Append the formatted data
            data.append({
                'sentence': sentence, # pronoun replaced with _
                'option1': option1,
                'option2': option2,
                'answer': str(entry['label'] + 1)  # Convert 0-indexed to 1-indexed
            })

    df = pd.DataFrame(data)
    for col in ['sentence', 'option1', 'option2', 'answer']:
        df[col] = df[col].astype(str)
    return df

def main() -> None:
    """Main function to execute the conversion."""
    args = parse_arguments()
    if args.output is None:
        args.output = args.input.with_suffix('.xlsx')  # Default output name

    xwinograd_df = convert_to_xwinograd(args.input)
    print(xwinograd_df.columns)
    # 
    xwinograd_df.to_excel(args.output, index=False)
    dataset = Dataset.from_pandas(xwinograd_df)
    dataset = DatasetDict({'test': dataset})

    logging.info(f'Successfully saved the converted dataset to {args.output}')
    login(new_session=True)
    DATASET_ID = "PolyAgent/xwinograd_uk"
    logging.info(f"Pushing dataset to hub as {DATASET_ID}")
    dataset.push_to_hub(DATASET_ID,private=True)

if __name__ == '__main__':
    main()