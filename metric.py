import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
def extract_metrics_from_json(folder_path):
    true_lst = []
    pred_lst = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                data = json.load(file)
                true_lst.append(data['True_Answer_List'])
                pred_lst.append(data['Final_Answer_List'])

    print(f"Extracted {len(true_lst)} entries from {folder_path}")
    return true_lst, pred_lst



def evaluate_metrics(true_lst, pred_lst):
    labels = ['Cardiotoxicity', 'Hematological', 'Infertility', 'Liver', 'Pulmonary', 'Renal']
    
    true_arr = np.array(true_lst).T
    pred_arr = np.array(pred_lst).T

    results = []

    for i, label in enumerate(labels):
        y_true = true_arr[i]
        y_pred = pred_arr[i]

        results.append([
            round(f1_score(y_true, y_pred), 4)
        ])

    metrics = ['F1']
    
    df = pd.DataFrame(results, columns=metrics, index=labels)
    df = df.T
    df.columns = labels
    
    return df


def evaluate_f1_score(true_lst, preds_lst):

    toxicity_names = [
        "Cardiotoxicity",
        "Dermatological Toxicity",
        "Hematological Toxicity",
        "Infertility",
        "Liver Toxicity",
        "Ototoxicity",
        "Pulmonary Toxicity",
        "Renal Toxicity"
    ]

    f1_scores_per_toxicity = {}

    for i, toxicity_name in enumerate(toxicity_names):
        true_values = [smiles[i] for smiles in true_lst]
        pred_values = [smiles[i] for smiles in preds_lst]
        
        f1_scores_per_toxicity[toxicity_name] = round(f1_score(true_values, pred_values, average='binary'),4)
    
    average_f1_score = round(sum(f1_scores_per_toxicity.values()) / len(toxicity_names), 4)
    f1_scores_per_toxicity["Average_F1_Score"] = average_f1_score

    return f1_scores_per_toxicity




