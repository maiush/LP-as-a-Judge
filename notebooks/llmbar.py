import os, pickle
import torch as t
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from lpaaj.data import LLMBar
from lpaaj.constants import RESULTS_DIR, GRAPH_PATH
from tqdm import tqdm


models = ["gemma-2-2b", "gemma-2-9b", "gemma-2-27b", "llama-3.1-8b", "llama-3.1-70b"]
model_titles = {
    "gemma-2-2b": "Gemma 2 2B",
    "gemma-2-9b": "Gemma 2 9B",
    "gemma-2-27b": "Gemma 2 27B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "llama-3.1-70b": "Llama 3.1 70B"
}

dirs = [f for f in os.listdir(RESULTS_DIR) if "llmbar" in f]
subsets = [f.split("-")[1] for f in dirs]

for model in models: 
    if os.path.exists(f"{GRAPH_PATH}/llmbar_{model}.png"):
        continue
    results = pd.DataFrame(columns=["subset", "method", "f1"])
    for subset in tqdm(subsets):
        # load data (and labels)
        data = LLMBar(
            subset=subset,
            task="compare",
        )
        # prompting
        preds = pickle.load(
            open(
                f"{RESULTS_DIR}/llmbar-{subset}/{model}/compare.pkl",
                "rb"
            )
        )
        score = f1_score(data.labels, preds, average="weighted", labels=[1, 2])
        results.loc[len(results)] = [subset, "pairwise-comparisons", score]
        # probing
        xpath = f"{RESULTS_DIR}/llmbar-{subset}/{model}/contrast"
        x1 = t.load(f"{xpath}_1.pt", weights_only=True).float()
        x2 = t.load(f"{xpath}_2.pt", weights_only=True).float()
        x1 -= x1.mean(0)
        x2 -= x2.mean(0)
        x = x1 - x2
        y = t.tensor(data.labels, dtype=int)
        mask = y != -1
        x, y = x[mask], y[mask]
        perm = t.randperm(len(x))
        x, y = x[perm], y[perm]
        split_ix = int(0.7*len(x))
        x_train, x_test = t.tensor_split(x, [split_ix], dim=0)
        y_train, y_test = t.tensor_split(y, [split_ix], dim=0)
        # supervised probe
        lr = LogisticRegression(
            solver="lbfgs",
            fit_intercept=False,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
        )
        lr.fit(x_train, y_train)
        preds = lr.predict(x_test)
        score = f1_score(y_test, preds, average="weighted", labels=[1, 2])
        results.loc[len(results)] = [subset, "s-probe", score]
        # unsupervised probe
        pca = PCA(1)
        pca.fit(x_train)
        preds = pca.transform(x_test).squeeze(1)
        p1 = t.tensor(preds > 0, dtype=t.int64) + 1
        p2 = t.tensor(preds < 0, dtype=t.int64) + 1
        score = max(
            f1_score(y_test, p1, average="weighted", labels=[1, 2]),
            f1_score(y_test, p2, average="weighted", labels=[1, 2])
        )
        results.loc[len(results)] = [subset, "u-probe", score]

    # Set figure size
    plt.figure(figsize=(15, 6))

    # Get unique subsets and methods
    subsets = results['subset'].unique()
    # Move Normal and Natural to front
    subset_list = list(subsets)
    for special in ['Natural', 'Normal']:
        if special in subset_list:
            subset_list.remove(special)
    subset_list = ['Normal', 'Natural'] + subset_list
    subsets = np.array(subset_list)

    methods = ['pairwise-comparisons', 's-probe', 'u-probe']

    # Set width of bars and positions of the bars
    width = 0.25
    x = np.arange(len(subsets))

    # Define stronger colors
    colors = ['#4a90d4', '#7ac17a', '#e67c73']  # Stronger blue, green, and pink

    # Create bars for each method
    for i, method in enumerate(methods):
        scores = [results[(results['subset'] == subset) & (results['method'] == method)]['f1'].values[0] 
                for subset in subsets]
        plt.bar(x + (i-1)*width, scores, width, label=method, color=colors[i])

    # Customize the plot
    plt.xlabel('Subset', fontsize=16)
    plt.ylabel('F1 Score', fontsize=16)
    plt.title(f'{model_titles[model]}: Performance Under Adversarial Prompting', fontsize=16)

    # Make Normal and Natural labels bold
    labels = [f'$\\mathbf{{{s}}}$' if s in ['Normal', 'Natural'] else s for s in subsets]
    plt.xticks(x, labels, rotation=25, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/llmbar_{model}.png", dpi=400)
    plt.close()