import pickle, yaml, json, re, gzip
from datetime import date
from tqdm import tqdm
from openai import OpenAI
from liquid import Template
import pandas as pd
import os
import signal
from threading import Timer

import pubchempy as pcp
import requests

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out.")

today = date.today()
formatted_date = today.strftime("%y%m%d") 

client = "Your_API_KEY"

system_prompt = '''You are an assistant designed to extract and filter toxicity-related information from a chemical's pathways and Gene Ontology (GO) terms. Your task is to return the filtered results as a JSON object with two keys: pathways and go_terms.
Ensure that the final output contains no duplicate entries. Only include unique entries that are directly related to toxicity.'''

user_prompt = Template("""
From the provided list of pathways and GO terms, extract only those related to toxicity. Follow these rules:
1. Include only entries directly or indirectly related to toxicity.
2. Ensure that the final lists contain **no duplicate entries** (only unique entries).

Pathway List: {{list_of_pathways}}
GO List: {{list_of_GO_terms}}

Expected Output:
'''json
{
    "pathways": ["list of pathways related to toxicity"],
    "go_terms": ["list of GO terms related to toxicity"]
}'''""")


def extract_tox_path_go(pathway_lst, GO_lst, max_retries=3, timeout=180):
    attempt = 0

    while attempt < max_retries:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt.render(list_of_pathways=pathway_lst, list_of_GO_terms=GO_lst)
                    }
                ],
                model="gpt-4o",
                temperature=0.0,
                seed=42
            )

            # Disable the alarm after the request succeeds
            signal.alarm(0)

            contents = chat_completion.choices[0].message.content
            if "```json" in contents and "```" in contents:
                start = contents.index("```json") + len("```json")
                end = contents.rindex("```")
                contents = contents[start:end].strip()

            extract_data = json.loads(contents)
            return extract_data

        except TimeoutError:
            print(f"Attempt {attempt + 1}: Request timed out. Retrying...")
            attempt += 1

        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}: Unable to parse JSON content. Retrying...")
            attempt += 1

        except Exception as e:
            print(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")
            attempt += 1

    print("Error: All attempts to process the response failed.")
    return None



if __name__ == "__main__":
    drug_list = ['entecavir']
    cell_lines = ['HEPG2', 'A549', 'HA1E']

    case_stduy_json = {}

    for d in drug_list:
        for cell_line in cell_lines:
            file_name = f"./{d}_{cell_line}.pkl"
            print(f"Processing {file_name}...")

            compound = pcp.get_compounds(d, 'name')[0]
            cid = compound.cid
            iupac_name = compound.iupac_name if compound.iupac_name else "N/A"

            with open(file_name, 'rb') as f:
                data = pickle.load(f)
            smiles = data['smiles']    
            path_lst = data['pathway']
            go_lst = data['go']
            extract_data = extract_tox_path_go(path_lst, go_lst)

            case_stduy_json[f"{d}_{cell_line}"] = {
                "SMILES": smiles,
                "pathways": extract_data['pathways'],
                "go_terms": extract_data['go_terms'],
                "iupac_name": iupac_name
            }


    output_file = 'case_study_entecavir.json'
    with open(output_file, 'w') as f:
        json.dump(case_stduy_json, f, indent=4)

    print(f"Test JSON saved to {output_file}")