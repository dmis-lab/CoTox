import json, os
from datetime import date
from tqdm import tqdm
from liquid import Template
import pandas as pd
import sys
import os

from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
]

google_search_tool = Tool(
    google_search = GoogleSearch()
)

saved_name = f"cotox_iupac_case_study"
client = "Your Google API KEY"

system_prompt = '''You are an expert in cheminformatics and toxicology. Your task is to predict organ-level toxicity for small molecules using:

1. Organ-specific pathway involvement in toxicity mechanisms.
2. Organ-specific GO terms' biological implications.
3. IUPAC name-based structural interpretation to support and explain toxicity mechanisms.

Your response must be strictly in JSON format. **Do not include any explanation, text, or information outside the JSON object.** The JSON format is as follows:'''

user_prompt = Template("""
Predict toxicity for each type ("Toxic" or "Non-Toxic") based on the provided **organ-specific** Pathways and GO Terms. Use IUPAC name analysis only to support the evidence from Pathway and GO Term analyses.
Finally, provide a step-by-step explanation of the overall mechanism combining evidence from Pathways, GO Terms, and the IUPAC name to describe how the compound causes toxicity in the body.

### IUPAC Name:
{{iupac_name}}
                       
### Liver Toxicity
- Pathway: {{liver_pathways}}
- GO Terms: {{liver_go}}
                       
### Pulmonary Toxicity
- Pathway: {{pulmonary_pathways}}
- GO Terms: {{pulmonary_go}}

### Renal Toxicity
- Pathway: {{renal_pathways}}
- GO Terms: {{renal_go}}


### Required Response:
- Return the response strictly in the JSON format below.
- Do not include any additional text, explanation, or comments outside the JSON.

```json
{
    "Summary": "A detailed overview that identifies potential toxicities based on Pathway and GO Term analyses, with IUPAC name used to explain structural contributions to the identified mechanisms.",
    "Toxicity Predictions": {
        "Liver Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in liver toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to liver toxicity based on GO Term analysis.",
                "IUPAC Support: Structural contributions from IUPAC name relevant to liver toxicity.",
                "Overall Mechanism: Combined explanation for liver toxicity."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },    
        "Pulmonary Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in pulmonary toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to pulmonary toxicity based on GO Term analysis.",
                "IUPAC Support: Structural contributions from IUPAC name relevant to pulmonary toxicity.",
                "Overall Mechanism: Combined explanation for pulmonary toxicity."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        },
        "Renal Toxicity": {
            "Reasoning": [
                "Pathway: Explanation of pathway involvement in renal toxicity and the biological processes it triggers.",
                "GO Term: Explanation of biological outcomes linked to renal toxicity based on GO Term analysis.",
                "IUPAC Support: Structural contributions from IUPAC name relevant to renal toxicity.",
                "Overall Mechanism: Combined explanation for renal toxicity."
            ],
            "Prediction": "Toxic" or "Non-Toxic"
        }
    }
}
```""")


def _build_chat_prompt(messages: list[dict[str,str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def tox_summary(idx, drug_name, iupac_name, pathway_lst, go_lst, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[_build_chat_prompt([
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt.render(
                            iupac_name=iupac_name,
                            liver_pathways=pathway_lst[0],
                            pulmonary_pathways=pathway_lst[1],
                            renal_pathways=pathway_lst[2],
                            liver_go=go_lst[0],
                            pulmonary_go=go_lst[1],
                            renal_go=go_lst[2]
                            
                        )
                    }
                ])],
                config=types.GenerateContentConfig(
                    candidate_count=1,
                    max_output_tokens=10000,
                    temperature=0.0,
                    safety_settings=safety_settings,
                    # tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            content = response.text
            
            if "```json" in content and "```" in content:
                start = content.index("```json") + len("```json")
                end = content.rindex("```")
                content = content[start:end].strip()

            toxicity_data = json.loads(content)

            os.makedirs(f"./results_case_study_entecavir/{saved_name}", exist_ok=True)

            with open(f"./results_case_study_entecavir/{saved_name}/{idx}_{drug_name}.json", "w") as json_file:
                json.dump(toxicity_data, json_file, indent=4)

            return toxicity_data

        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}: Unable to parse JSON content. Retrying...")
            attempt += 1

        except Exception as e:
            print(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")
            attempt += 1

    print("Error: All attempts to process the response failed.")
    return None

file_path = "case_study_entecavir.json"
with open(file_path, "r") as file:
    case_study_data = json.load(file)

case_df = pd.read_csv('case_study_entecavir.csv', index_col=False)

if __name__ == "__main__":
    print(f"Processing {saved_name}")
    preds_lst = []

    for i in tqdm(range(len(case_df))):
        chemical_name = case_df['cmap_name'].tolist()[i]
        exp_key = f'{chemical_name}_A549'

        iupac_name = case_study_data[exp_key]["iupac_name"]
        path_lst = [
            case_study_data[key]["pathways"]
            for key in case_study_data
            if chemical_name.lower() in key.lower()
        ]
        go_lst = [
            case_study_data[key]["go_terms"]
            for key in case_study_data
            if chemical_name.lower() in key.lower()
        ]

        tox_pred = tox_summary(i, chemical_name, iupac_name, path_lst, go_lst)


    print("All Done")