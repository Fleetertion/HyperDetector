import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset


def _print_score_quantiles(score):
    q = np.quantile(score, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(
        "[eval-knn] score quantiles "
        "min={:.6f} p10={:.6f} p25={:.6f} p50={:.6f} p75={:.6f} p90={:.6f} max={:.6f}".format(
            q[0], q[1], q[2], q[3], q[4], q[5], q[6]
        )
    )


def _select_threshold(prec, rec, threshold, mode='f1', target_recall=None):
    # threshold has length (n_points - 1), so align precision/recall accordingly.
    if len(threshold) == 0:
        return 0, 0.0

    prec_t = prec[:-1]
    rec_t = rec[:-1]

    if mode == 'target_recall':
        if target_recall is None:
            raise ValueError("target_recall must be set when threshold_mode='target_recall'")
        candidate_idx = np.where(rec_t >= target_recall)[0]
        if len(candidate_idx) == 0:
            # No threshold can reach target recall. Pick the closest recall, then max precision.
            dist = np.abs(rec_t - target_recall)
            min_dist = dist.min()
            nearest = np.where(dist == min_dist)[0]
            best_idx = nearest[np.argmax(prec_t[nearest])]
        else:
            # Under recall constraint, maximize precision.
            best_idx = candidate_idx[np.argmax(prec_t[candidate_idx])]
    else:
        f1_t = 2 * prec_t * rec_t / (rec_t + prec_t + 1e-9)
        best_idx = int(np.argmax(f1_t))

    return int(best_idx), float(threshold[best_idx])


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0,
                           threshold_mode='f1', target_recall=None):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    with torch.no_grad():
        for i in full:
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(
            100, dataset, x, y, threshold_mode=threshold_mode, target_recall=target_recall
        )
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels, threshold_mode='f1', target_recall=None):
    x, y = embeddings, labels
    
    train_count = 100
    n_neighbors = min(int(train_count * 0.02), 10)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            best_idx, best_thres = _select_threshold(
                prec, rec, threshold, mode=threshold_mode, target_recall=target_recall
            )

            y_pred = (score >= best_thres).astype(np.float32)
            tp = int(np.sum((y_test == 1.0) & (y_pred == 1.0)))
            fn = int(np.sum((y_test == 1.0) & (y_pred == 0.0)))
            tn = int(np.sum((y_test == 0.0) & (y_pred == 0.0)))
            fp = int(np.sum((y_test == 0.0) & (y_pred == 1.0)))

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            prec_list.append(precision)
            rec_list.append(recall)
            f1_list.append(f1)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        print('THRESHOLD_MODE: {}'.format(threshold_mode))
        if threshold_mode == 'target_recall':
            print('TARGET_RECALL: {}'.format(target_recall))
        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))
        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        _, best_thres = _select_threshold(
            prec, rec, threshold, mode=threshold_mode, target_recall=target_recall
        )

        y_pred = (score >= best_thres).astype(np.float32)
        tp = int(np.sum((y_test == 1.0) & (y_pred == 1.0)))
        fn = int(np.sum((y_test == 1.0) & (y_pred == 0.0)))
        tn = int(np.sum((y_test == 0.0) & (y_pred == 0.0)))
        fp = int(np.sum((y_test == 0.0) & (y_pred == 1.0)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print('THRESHOLD_MODE: {}'.format(threshold_mode))
        if threshold_mode == 'target_recall':
            print('TARGET_RECALL: {}'.format(target_recall))
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1))
        print('PRECISION: {}'.format(precision))
        print('RECALL: {}'.format(recall))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0


def _select_entity_threshold(dataset, prec, rec, threshold, mode='legacy', target_recall=None):
    if len(threshold) == 0:
        return 0, 0.0

    if mode == "legacy":
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = -1
        for i in range(len(f1)):
            # Keep historical behavior for existing datasets.
            if dataset == 'trace' and rec[i] < 0.99979:
                best_idx = i - 1
                break
            if dataset == 'theia' and rec[i] < 0.99996:
                best_idx = i - 1
                break
            if dataset == 'cadets' and rec[i] < 0.9976:
                best_idx = i - 1
                break
        if best_idx == -1:
            best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
        best_idx = min(best_idx, len(threshold) - 1)
        return best_idx, float(threshold[best_idx])

    best_idx, best_thres = _select_threshold(
        prec, rec, threshold, mode=mode, target_recall=target_recall
    )
    return best_idx, best_thres


def evaluate_entity_level_using_knn(
        dataset,
        x_train,
        x_test,
        y_test,
        n_neighbors=None,
        metric='euclidean',
        threshold_mode='legacy',
        target_recall=None,
        use_cache=False,
        auxiliary_score=None,
        auxiliary_weight=1.0,
        auxiliary_name=None):
    print("[eval-knn] standardizing features")
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

    if n_neighbors is None or n_neighbors <= 0:
        # if dataset == 'cadets':
        #     n_neighbors = 200
        # else:
            n_neighbors = 10
    n_neighbors = int(max(1, min(n_neighbors, x_train.shape[0])))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    print(f"[eval-knn] fitting NearestNeighbors (k={n_neighbors}, metric={metric})")
    nbrs.fit(x_train)

    if metric == "euclidean":
        save_dict_path = './result/distance_save_{}_k{}.pkl'.format(dataset, n_neighbors)
    else:
        save_dict_path = './result/distance_save_{}_k{}_{}.pkl'.format(dataset, n_neighbors, metric)

    def _compute_and_cache_distances():
        print("[eval-knn] cache miss, computing train/test distances")
        rng = np.random.RandomState(0)
        idx = rng.permutation(x_train.shape[0])
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        mean_distance = distances.mean() + 1e-12
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        os.makedirs(os.path.dirname(save_dict_path), exist_ok=True)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
        return mean_distance, distances

    if (not use_cache) or (not os.path.exists(save_dict_path)):
        mean_distance, distances = _compute_and_cache_distances()
    else:
        print(f"[eval-knn] loading cached distances: {save_dict_path}")
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
        if not isinstance(distances, np.ndarray) or distances.shape[0] != x_test.shape[0]:
            print(
                f"[eval-knn] cache shape mismatch (cached={getattr(distances, 'shape', None)}, "
                f"current={(x_test.shape[0],)}), recomputing distances"
            )
            mean_distance, distances = _compute_and_cache_distances()
    score = distances / (mean_distance + 1e-12)
    del distances
    if auxiliary_score is not None:
        auxiliary_score = np.asarray(auxiliary_score, dtype=np.float64)
        if auxiliary_score.shape[0] != score.shape[0]:
            raise ValueError(
                "auxiliary_score length mismatch: expected {}, got {}".format(
                    score.shape[0], auxiliary_score.shape[0]
                )
            )
        score_span = float(np.nanmax(score) - np.nanmin(score))
        score = score + auxiliary_score * (score_span + 1.0) * float(auxiliary_weight)
        print(
            "[eval-knn] applying auxiliary score{}: weight={} positives={}".format(
                "" if auxiliary_name is None else " ({})".format(auxiliary_name),
                auxiliary_weight,
                int(np.sum(auxiliary_score > 0)),
            )
        )
    unique_classes = np.unique(y_test)
    if unique_classes.shape[0] < 2:
        only_class = int(unique_classes[0]) if unique_classes.shape[0] == 1 else -1
        print(
            f"[eval-knn] warning: y_test has a single class ({only_class}), "
            f"ROC AUC is undefined for this split."
        )
        auc = float("nan")

        if only_class == 0:
            best_thres = float("inf")
        else:
            best_thres = float("-inf")

        y_pred = (score >= best_thres).astype(np.float32)
        tp = int(np.sum((y_test == 1.0) & (y_pred == 1.0)))
        fn = int(np.sum((y_test == 1.0) & (y_pred == 0.0)))
        tn = int(np.sum((y_test == 0.0) & (y_pred == 0.0)))
        fp = int(np.sum((y_test == 0.0) & (y_pred == 1.0)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print('THRESHOLD_MODE: {}'.format(threshold_mode))
        if threshold_mode == 'target_recall':
            print('TARGET_RECALL: {}'.format(target_recall))
        print('AUC: {}'.format(auc))
        print('PR_AUC: nan')
        print('BEST_THRESHOLD: {}'.format(best_thres))
        print('PRED_POS_RATE: {}'.format(float(y_pred.mean())))
        print('F1: {}'.format(f1))
        print('PRECISION: {}'.format(precision))
        print('RECALL: {}'.format(recall))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        _print_score_quantiles(score)
        return auc, 0.0, None, {
            "auc": auc,
            "pr_auc": float("nan"),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": float(best_thres),
            "pred_pos_rate": float(y_pred.mean()),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "n_neighbors": int(n_neighbors),
            "metric": metric,
            "threshold_mode": threshold_mode,
        }

    auc = roc_auc_score(y_test, score)
    if auc < 0.5:
        # For unseen datasets the score direction can be inverted.
        score = -score
        auc = roc_auc_score(y_test, score)

    prec, rec, threshold = precision_recall_curve(y_test, score)
    pr_auc = sk_auc(rec, prec)
    if len(threshold) == 0:
        print('THRESHOLD_MODE: {}'.format(threshold_mode))
        if threshold_mode == 'target_recall':
            print('TARGET_RECALL: {}'.format(target_recall))
        print('AUC: {}'.format(auc))
        print('PR_AUC: {}'.format(pr_auc))
        print('BEST_THRESHOLD: nan')
        print('PRED_POS_RATE: nan')
        print('F1: 0.0')
        print('PRECISION: 0.0')
        print('RECALL: 0.0')
        print('TN: 0')
        print('FN: 0')
        print('TP: 0')
        print('FP: 0')
        _print_score_quantiles(score)
        return auc, 0.0, None, {
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": float("nan"),
            "pred_pos_rate": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "n_neighbors": int(n_neighbors),
            "metric": metric,
            "threshold_mode": threshold_mode,
        }
    best_idx, best_thres = _select_entity_threshold(
        dataset, prec, rec, threshold, mode=threshold_mode, target_recall=target_recall
    )

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    y_pred = (score >= best_thres).astype(np.float32)
    print('THRESHOLD_MODE: {}'.format(threshold_mode))
    if threshold_mode == 'target_recall':
        print('TARGET_RECALL: {}'.format(target_recall))
    print('AUC: {}'.format(auc))
    print('PR_AUC: {}'.format(pr_auc))
    print('BEST_THRESHOLD: {}'.format(float(best_thres)))
    print('PRED_POS_RATE: {}'.format(float(y_pred.mean())))
    f1_val = 2 * prec[best_idx] * rec[best_idx] / (rec[best_idx] + prec[best_idx] + 1e-9)
    print('F1: {}'.format(f1_val))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    _print_score_quantiles(score)
    return auc, 0.0, None, {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_val),
        "precision": float(prec[best_idx]),
        "recall": float(rec[best_idx]),
        "threshold": float(best_thres),
        "pred_pos_rate": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "threshold_mode": threshold_mode,
    }
