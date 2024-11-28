import torch
from torch import nn
from datahandling import get_dev_data
from tqdm import tqdm
import config 
from models.vqa_model import VQA
from os.path import join
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datahandling import load_multilabel_binarizer

import pandas as pd


# Change this to the filename of the saved model in the models/trained folder
# MODEL_NAME = "vqa_model_dinov2.pth"
# model_id = "facebook/dinov2-large"


# Dont change this unless there are mor answers possible than before
mlb = load_multilabel_binarizer()
NUM_LABELS = len(mlb.classes_)

    
def build_table_with_metrics_per_question_and_summary(y_true, y_pred, model_name):
    idx_without = []

    # save which indices of y_true/y_preds are for the questions
    per_question = dict()

    results = {
        "question": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for answer in mlb.classes_:
        question = answer.split("_")[0]
        per_question[question] = []
        
        
    for idx, truth in enumerate(y_true):
        correct_answers = mlb.inverse_transform(np.array([truth]))[0]

        if len(correct_answers) > 0:
            first_answer = correct_answers[0]
            key = first_answer.split("_")[0]
            per_question[key].append(idx)
        else:
            idx_without.append(idx)
    # from y_preds, y_true get the indices per question, then calculate the metrics on these instances

    for question, indexes in per_question.items():
        results["question"].append(question)
        results["accuracy"].append(round(accuracy_score(y_true[indexes], y_pred[indexes],4)))
        results["precision"].append(round(precision_score(y_true[indexes], y_pred[indexes], average="samples", zero_division=0), 4))
        results["recall"].append(round(recall_score(y_true[indexes], y_pred[indexes], average="samples", zero_division=0), 4))
        results["f1"].append(round(f1_score(y_true[indexes], y_pred[indexes], average="samples", zero_division=0), 4))
        
    print(f"Nr of invalid answers: {idx_without.__len__()} ")
    
    df = pd.DataFrame.from_dict(results)
    print(df)
    latex_table = df.to_latex(index=False, float_format="%.4f")


    summary_stats = """ 
    \\hline
    \\textbf{All} &  """ + f" \\textbf{{{df.accuracy.mean():.4f}}}  & \\textbf{{{df.precision.mean():.4f}}} & \\textbf{{{df.recall.mean():.4f}}} & \\textbf{{{df.f1.mean():.4f}}}" + """
    \\hline
    """

    insertion_point = latex_table.rfind(r"\bottomrule")
    latex_table = latex_table[:insertion_point] + summary_stats + latex_table[insertion_point:]

    new_header = r"""
    \begin{tabular}{lcccc}
    \toprule
    \textbf{question} & \textbf{accuracy} & \textbf{precision} & \textbf{recall} & \textbf{f1} \\
    \hline
    \midrule
    """

    midrule_index = latex_table.find(r"\midrule")
    modified_table = new_header + latex_table[midrule_index + len(r"\midrule"):]


    with open(f"results_dev_test_{model_name.split(".")[0]}.txt", "w") as f:
        f.write(modified_table)
    



def load_model(device, model_id, model_name) -> torch.nn.Module:
    model = VQA(vision_model_id=model_id)
    model.load_state_dict(torch.load(join(config.trained_model_path, model_name), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_test(model_name="vqa_aimv2.pth", model_id="apple/aimv2-large-patch14-224"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = load_model(device, model_id, model_name)
    _, _, test_loader = get_dev_data(image_processor_model_id=model_id, debug_mode=False)  
    criterion = nn.BCEWithLogitsLoss()

    loss = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            (images, questions, question_attention_mask), labels = batch
            images, questions, question_attention_mask, labels = images.to(device), questions.to(device), question_attention_mask.to(device), labels.to(device)
            
            outputs = model(images, questions, question_attention_mask)
            loss = criterion(outputs, labels)
            loss += loss.item()
            
            y_pred.append(outputs)
            y_true.append(labels)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    
    y_pred = torch.sigmoid(y_pred).detach().cpu() >= 0.5  # Threshold predictions
    y_true = y_true.detach().cpu()
    
    results = f""" Performance on Dev test set
    accuracy: {accuracy_score(y_true, y_pred):.4f},  
    precision: {precision_score(y_true, y_pred, average="samples", zero_division=0):.4f},  
    recall: {recall_score(y_true, y_pred, average="samples", zero_division=0):.4f}
    F1 Score: {f1_score(y_true, y_pred, average="samples", zero_division=0):.4f},  
    """
    print(results)
    
    build_table_with_metrics_per_question_and_summary(y_true, y_pred, model_name)