import torch, pickle
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def build_memory_bank(model, dataset, device="cpu"):
    embeds = []
    for data in dataset:
        data = data.to(device)
        with torch.no_grad():
            h, _ = model(data.x, data.incidence)
            embeds.append(h.mean(0).cpu())
    bank = torch.stack(embeds)
    kdt = KDTree(bank.numpy())
    return kdt, bank

def anomaly_scores(kdt: 'KDTree', emb_mat: torch.Tensor, k=30):
    """Return mean distance of each vector to its k nearest neighbours."""
    emb_mat = emb_mat.detach().cpu()        # ← 先切断 grad、移到 CPU
    d, _    = kdt.query(emb_mat.numpy(), k=k)
    return d.mean(axis=1)

def evaluate(scores, labels, fpr_target=0.05):
    # threshold chosen on benign subset
    benign_scores = [s for s, l in zip(scores, labels) if l == 0]
    th = np.percentile(benign_scores, 100*(1-fpr_target))
    pred = [int(s > th) for s in scores]
    auc = roc_auc_score(labels, scores)
    P,R,F,_ = precision_recall_fscore_support(labels, pred, average='binary')
    return {"AUC": auc, "P": P, "R": R, "F1": F, "thr": th}
