import json, gzip, os, pickle, time, math
from datetime import date
from tqdm import tqdm
from openai import OpenAI
from liquid import Template
import argparse
import pandas as pd
from metric import *

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser example")
    parser.add_argument('--test_name', type=str, default='test_1')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--error', type=bool, default=False)
    return parser.parse_args()

args = parse_args()
test_name = args.test_name
unitox_df = pd.read_csv(f'./CTD/Unitox_CTD_Drug_{test_name}.csv')

toxicity_columns = [
    col for col in unitox_df.columns
    if col.endswith("_binary_rating_0_1")
    and not any(x in col for x in ['dermatological', 'ototoxicity'])
]

true_lst = unitox_df[toxicity_columns].fillna(0).astype(int).values.tolist()
toxicity_types = [
        "Cardiotoxicity",
        "Hematological Toxicity",
        "Infertility",
        "Liver Toxicity",
        "Pulmonary Toxicity",
        "Renal Toxicity"
    ]
saved_name = f"cotox_iupac_gpt_4o_{test_name}"

client = "Your_API_KEY"

system_prompt = '''You are an expert in cheminformatics and toxicology. Your task is to predict toxicity for small molecules using:

1. Pathway involvement in toxicity mechanisms.
2. GO terms' biological implications.
3. IUPAC name-based structural interpretation to support and explain toxicity mechanisms.

Your response must be strictly in JSON format. **Do not include any explanation, text, or information outside the JSON object.** The JSON format is as follows:'''

user_prompt = Template("""
Predict toxicity for each type ("Toxic" or "Non-Toxic") based on the provided **organ-specific** Pathways and GO Terms. Use IUPAC name analysis only to support the evidence from Pathway and GO Term analyses.
Finally, provide a step-by-step explanation of the overall mechanism combining evidence from Pathways, GO Terms, and the IUPAC name to describe how the compound causes toxicity in the body.

### IUPAC Name:
{{iupac_name}}

### List of Pathway:
{{pathway_lst}}

### List of GO Term:
{{GO_lst}}

### Required Response:
- Return the response strictly in the JSON format below.
- Do not include any additional text, explanation, or comments outside the JSON.

```json
{
    "Summary": "A detailed overview that identifies potential toxicities based on Pathway and GO Term analyses, with IUPAC name used to explain structural contributions to the identified mechanisms.",
    "Toxicity Predictions": {
        "Cardiotoxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in cardiotoxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to cardiotoxicity based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to cardiotoxicity.",
                "Overall Mechanism: Combined explanation of how the compound causes cardiotoxicity in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Hematological Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in hematological toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to hematological toxicity based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to hematological toxicity.",
                "Overall Mechanism: Combined explanation of how the compound causes hematological toxicity in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Infertility": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in infertility and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to infertility based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to infertility.",
                "Overall Mechanism: Combined explanation of how the compound causes infertility in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Liver Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in liver toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to liver toxicity based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to liver toxicity.",
                "Overall Mechanism: Combined explanation of how the compound causes liver toxicity in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Pulmonary Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in pulmonary toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to pulmonary toxicity based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to pulmonary toxicity.",
                "Overall Mechanism: Combined explanation of how the compound causes pulmonary toxicity in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Renal Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in renal toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to renal toxicity based on GO Term analysis.",
                "IUPAC Support: Explanation of how structural features inferred from the IUPAC name contribute to the biological processes and pathways leading to renal toxicity.",
                "Overall Mechanism: Combined explanation of how the compound causes renal toxicity in the body."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        }
    }
}
```""")



def tox_summary(idx, drug_name, iupac_name, pathway_lst, go_lst, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt.render(
                            iupac_name=iupac_name,
                            pathway_lst=pathway_lst,
                            GO_lst=go_lst
                            
                        )
                    }
                ],
                model="gpt-4o",
                logprobs=True,
                temperature=0.0,
                seed=42
            )

            content = chat_completion.choices[0].message.content
            if "```json" in content and "```" in content:
                start = content.index("```json") + len("```json")
                end = content.rindex("```")
                content = content[start:end].strip()
            
            toxicity_data = json.loads(content)
            final_answer_list = []
            for toxicity_type in toxicity_types:
                pred_str = toxicity_data["Toxicity Predictions"][toxicity_type]["Prediction"]
                if pred_str == "Toxic":
                    final_answer_list.append(1)
                else:
                    final_answer_list.append(0)
            toxicity_data["Final_Answer_List"] = final_answer_list
            toxicity_data["True_Answer_List"] = true_lst[idx]

            os.makedirs(f"./results/{saved_name}", exist_ok=True)

            with open(f"./results/{saved_name}/{idx}_{drug_name}.json", "w") as json_file:
                json.dump(toxicity_data, json_file, indent=4)

            return toxicity_data['Final_Answer_List']

        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}: Unable to parse JSON content. Retrying...")
            attempt += 1

        except Exception as e:
            print(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")
            attempt += 1

    print("Error: All attempts to process the response failed.")
    return None


file_path = f"Unitox_CTD_Drug_{test_name}.json"
with open(file_path, "r") as file:
    data = json.load(file)

if __name__ == "__main__":
    print(f"Processing {saved_name}")
    preds_lst = []
    for i in tqdm(range(len(unitox_df))):
        Chemical_name = unitox_df['generic_name'].tolist()[i].lower()
        smiles = unitox_df['smiles'].tolist()[i]
        key_name = f"{i}_{Chemical_name}"
        iupac_name = data[key_name]['iupac_name']
        path_lst = list(set(data[key_name]['pathways']))
        GO_lst = list(set(data[key_name]['go_terms']))

        tox_pred = tox_summary(i, Chemical_name, iupac_name, path_lst, GO_lst)

        preds_lst.append(tox_pred)

    metric_results_df= evaluate_metrics(true_lst, preds_lst)

    output_file = f"./results/{saved_name}/scores_output.csv"                               
    metric_results_df.to_csv(output_file)

    print(f"{saved_name} Scores Saved!!")