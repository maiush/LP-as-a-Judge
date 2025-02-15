from lpaaj.constants import RESULTS_DIR
from lpaaj.data import MTBench

import pickle
import torch as t
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from tqdm import trange


def train_supervised(x_train, x_test, y_train, y_test):
    n_layers = x_train.shape[1]
    scores = []
    probes = []
    for layer in trange(n_layers, desc="fitting probes"):
        lr = LogisticRegression(
            solver="lbfgs",
            fit_intercept=False,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
        )
        lr.fit(x_train[:, layer], y_train)
        preds = lr.predict(x_test[:, layer])
        score = f1_score(y_test, preds, average="weighted", labels=[1, 2])
        scores.append(score)
        probes.append(lr.coef_[0])
    return scores, probes

def train_unsupervised(x_train, x_test, y_train, y_test):
    n_layers = x_train.shape[1]
    scores = []
    probes = []
    for layer in trange(n_layers, desc="fitting probes"):
        pca = PCA(1)
        pca.fit(x_train[:, layer])
        preds = pca.transform(x_test[:, layer]).squeeze(1)
        p1 = t.tensor(preds > 0, dtype=t.int64) + 1
        p2 = t.tensor(preds < 0, dtype=t.int64) + 1
        score1 = f1_score(y_test, p1, average="weighted", labels=[1, 2])
        score2 = f1_score(y_test, p2, average="weighted", labels=[1, 2])
        if score1 > score2:
            scores.append(score1)
            probes.append(pca.components_[0])
        else:
            scores.append(score2) 
            probes.append(-pca.components_[0])
    return scores, probes

def main(model: str) -> None:
    data = MTBench(task="compare")
    x1 = t.load(f"{RESULTS_DIR}/mtbench/{model}/contrast_1_all.pt", weights_only=True).float()
    x2 = t.load(f"{RESULTS_DIR}/mtbench/{model}/contrast_2_all.pt", weights_only=True).float()
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
    s_scores, s_probes = train_supervised(x_train, x_test, y_train, y_test)
    u_scores, u_probes = train_unsupervised(x_train, x_test, y_train, y_test)
    with open(f"{RESULTS_DIR}/mtbench/{model}/s-scores.pkl", "wb") as f:
        pickle.dump(s_scores, f)
    with open(f"{RESULTS_DIR}/mtbench/{model}/s-probes.pkl", "wb") as f:
        pickle.dump(s_probes, f)
    with open(f"{RESULTS_DIR}/mtbench/{model}/u-scores.pkl", "wb") as f:
        pickle.dump(u_scores, f)
    with open(f"{RESULTS_DIR}/mtbench/{model}/u-probes.pkl", "wb") as f:
        pickle.dump(u_probes, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.model)