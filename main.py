
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr
import numpy as np

LABELS = ["Feature Specificity (FS)", "Rationale Clarity (RC)", "Problem oriented", "Language Clarity (LC)", "Internal Consistency (IC)"]


def normalize_by_equalizing(df1, df2):
        df1, df2 = df1.copy(), df2.copy()
        flag = True
        for label in LABELS:
            for i in range(len(df1)):
                if (df1[label][i] == 2 and df2[label][i] == 3) or (df1[label][i] == 3 and df2[label][i] == 2):
                    new_value = 2 if flag else 3
                    flag = not flag
                    df1.at[i, label] = new_value
                    df2.at[i, label] = new_value
        return df1, df2

def normalize_by_offset(df1, df2, offset=1):
    df1, df2 = df1.copy()[LABELS], df2.copy()[LABELS]
    for label in LABELS:
        for i in range(len(df1)):
            for df in [df1, df2]:
                if (df[label][i] != 1):
                    df.at[i, label] = df.at[i, label] + offset
    return df1, df2

def get_values_by_column(dataframes):
    values_by_columns = {}
    for label in LABELS:
        values_by_columns[label] = [df[label] for df in dataframes]
    return values_by_columns

def get_all_values(dataframes):
    all_labels_1 = []
    all_labels_2 = []
    for label in LABELS:
        all_labels_1.extend(dataframes[0][label].values)
        all_labels_2.extend(dataframes[1][label].values)
    return all_labels_1, all_labels_2

def compute_score(score_name, dataframes, f_indicator, f_normalize=lambda x, y: [x, y], log=True):

    normalized_dfs = f_normalize(*dataframes)
    scores_by_column = {}
    values_by_column = get_values_by_column(normalized_dfs)

    for label in LABELS:
        values = values_by_column[label]
        scores_by_column[label] = f_indicator(values[0], values[1])
        
    overall_score = f_indicator(*get_all_values(normalized_dfs))

    if log:
        print_score(score_name, scores_by_column, overall_score)

    
    return scores_by_column, overall_score


def pretty_score(score):
    descriptions = {(-2**10, 0): "Poor", (0, 0.2) : "Slight", (0.2, 0.4) : "Fair", (0.4, 0.6): "Moderate", (0.6, 0.8): "Substantial", (0.8, 2**10) : "Almost Perfect"}
    for (l,h) in descriptions:
        if score >= l and score < h:
            return descriptions[(l,h)]

def print_score(score_name, scores_by_column, overall_score):
    print("***")
    print(f"{score_name} by column:")
    for label, score in scores_by_column.items():
        print(f"{label}: {pretty_score(score)} ({score:.2f})")

    print(f"\nOverall {score_name} score: {pretty_score(overall_score)} ({overall_score:.2f})\n")


def weighted_kappa(x, y, weights):
    num_classes = 3
    o_matrix = np.zeros((num_classes, num_classes))
    for a, b in zip(x, y):
        o_matrix[a-1, b-1] += 1
    
    row_marginals = np.sum(o_matrix, axis=1)
    col_marginals = np.sum(o_matrix, axis=0)
    total = np.sum(o_matrix)
    e_matrix = np.outer(row_marginals, col_marginals) / total
    
    numerator = np.sum(weights * o_matrix)
    denominator = np.sum(weights * e_matrix)
    kappa = 1 - (numerator / denominator)
    return kappa



files = ['LLM User Stories - GQ.csv', 'LLM User Stories - LP.csv']  
dataframes = [pd.read_csv(file) for file in files]

compute_score("Cohen's Kappa", dataframes, cohen_kappa_score)
weights = [
    [0, 1, 1], 
    [1, 0, 0], 
    [1, 0, 0]   
]
compute_score("Cohen's Kappa (normalized data)", dataframes, cohen_kappa_score, normalize_by_equalizing)
weights = [
    [0, 1, 1], 
    [1, 0, 0], 
    [1, 0, 0]   
]
compute_score("Weighted Kappa", dataframes, lambda x,y: weighted_kappa(x, y, weights=weights))
compute_score("Spearman", dataframes, lambda x,y: spearmanr(x, y).statistic, lambda df1, df2: normalize_by_offset(df1, df2, offset=2))
compute_score("Pearson", dataframes, lambda x,y: pearsonr(x, y).statistic, lambda df1, df2: normalize_by_offset(df1, df2, offset=2))
