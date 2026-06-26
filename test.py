import torch
import warnings
import time
import re
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from utils.utils import set_random_seed
import numpy as np
from utils.config import build_args
from utils.apt2021_pipeline import apply_apt2021_cli_overrides, evaluate_apt2021
warnings.filterwarnings('ignore')


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _infer_num_layers_from_state_dict(state_dict):
    pattern = re.compile(r"^encoder\.gats\.(\d+)\.")
    layer_ids = []
    for k in state_dict.keys():
        match = pattern.match(k)
        if match is not None:
            layer_ids.append(int(match.group(1)))
    if len(layer_ids) == 0:
        return None
    return max(layer_ids) + 1


def _infer_num_hidden_from_state_dict(state_dict):
    weight = state_dict.get("encoder_to_decoder.weight")
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[0])
    return None


def _apply_dataset_model_defaults(main_args, dataset_name):
    if dataset_name == 'wget':
        default_hidden = 32
        default_layers = 4
    elif dataset_name == 'apt2021':
        default_hidden = 64
        default_layers = 2
    elif dataset_name == 'clearscope':
        default_hidden = 64
        default_layers = 3
        if main_args.entity_knn_k <= 0:
            main_args.entity_knn_k = 5
        if main_args.entity_knn_metric is None:
            main_args.entity_knn_metric = "cosine"
        if main_args.entity_use_cache is None:
            main_args.entity_use_cache = True
        main_args.entity_threshold_mode = main_args.entity_threshold_mode or "legacy"
    else:
        default_hidden = 64
        default_layers = 3
    if main_args.entity_knn_metric is None:
        main_args.entity_knn_metric = "euclidean"
    if main_args.entity_use_cache is None:
        main_args.entity_use_cache = False
    main_args.num_hidden = default_hidden if main_args.num_hidden is None else int(main_args.num_hidden)
    main_args.num_layers = default_layers if main_args.num_layers is None else int(main_args.num_layers)


def _align_model_args_to_checkpoint(main_args, state_dict):
    ckpt_layers = _infer_num_layers_from_state_dict(state_dict)
    ckpt_hidden = _infer_num_hidden_from_state_dict(state_dict)
    if ckpt_layers is not None and ckpt_layers != main_args.num_layers:
        print(
            f"[eval] checkpoint architecture mismatch detected: "
            f"num_layers={main_args.num_layers} -> {ckpt_layers}"
        )
        main_args.num_layers = ckpt_layers
    if ckpt_hidden is not None and ckpt_hidden != main_args.num_hidden:
        print(
            f"[eval] checkpoint architecture mismatch detected: "
            f"num_hidden={main_args.num_hidden} -> {ckpt_hidden}"
        )
        main_args.num_hidden = ckpt_hidden


def _parse_csv_int_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if text == "":
        return []
    result = []
    for token in text.split(","):
        token = token.strip()
        if token == "":
            continue
        result.append(int(token))
    return result


def _parse_csv_str_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if text == "":
        return []
    result = []
    for token in text.split(","):
        token = token.strip().lower()
        if token == "":
            continue
        result.append(token)
    return result


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset.lower()
    main_args.dataset = dataset_name
    _apply_dataset_model_defaults(main_args, dataset_name)
    set_random_seed(0)

    if dataset_name == 'wget':
        from model.autoencoder import build_model
        from utils.poolers import Pooling
        from model.eval import batch_level_evaluation

        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        checkpoint_path = "./result/checkpoint-{}.pt".format(dataset_name)
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = _normalize_state_dict_keys(state_dict)
        _align_model_args_to_checkpoint(main_args, state_dict)
        model = build_model(main_args)
        model.load_state_dict(state_dict)
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(
            model, pooler, device, ['knn'], main_args.dataset, main_args.n_dim, main_args.e_dim,
            threshold_mode=main_args.threshold_mode, target_recall=main_args.target_recall
        )
    else:
        if dataset_name == 'apt2021':
            apply_apt2021_cli_overrides(main_args, update_train=False, update_eval=True)
            test_auc, test_std = evaluate_apt2021(device=device)
            print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
            return
        from model.autoencoder import build_model
        from model.eval import evaluate_entity_level_using_knn

        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        checkpoint_path = "./result/checkpoint-{}.pt".format(dataset_name)
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = _normalize_state_dict_keys(state_dict)
        _align_model_args_to_checkpoint(main_args, state_dict)
        model = build_model(main_args)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        malicious_field = metadata.get('malicious', [])
        if isinstance(malicious_field, (list, tuple)) and len(malicious_field) == 2:
            malicious = malicious_field[0]
        else:
            malicious = malicious_field
        n_train = metadata['n_train']
        n_test = metadata['n_test']
        exclude_train_nodes_from_test = metadata.get('exclude_train_nodes_from_test', True)

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                step_start = time.time()
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                step_elapsed = time.time() - step_start
                print(
                    f"[eval-embed-train] graph={i + 1}/{n_train} "
                    f"nodes={g.num_nodes()} edges={g.num_edges()} time={step_elapsed:.2f}s"
                )
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                step_start = time.time()
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if exclude_train_nodes_from_test and i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                step_elapsed = time.time() - step_start
                print(
                    f"[eval-embed-test] graph={i + 1}/{n_test} "
                    f"nodes={g.num_nodes()} edges={g.num_edges()} time={step_elapsed:.2f}s"
                )
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            if exclude_train_nodes_from_test:
                # Exclude training samples from the test set
                test_idx = []
                for i in range(x_test.shape[0]):
                    if i >= skip_benign or y_test[i] == 1.0:
                        test_idx.append(i)
                result_x_test = x_test[test_idx]
                result_y_test = y_test[test_idx]
            else:
                result_x_test = x_test
                result_y_test = y_test
            del x_test, y_test
            auxiliary_score = None
            if dataset_name == 'cadets' and main_args.cadets_exec_context:
                from utils.cadets_context import load_or_build_cadets_exec_context_score

                auxiliary_full, auxiliary_details = load_or_build_cadets_exec_context_score(n)
                if auxiliary_full is None:
                    print("[eval-cadets-context] disabled: {}".format(auxiliary_details.get("reason")))
                else:
                    if exclude_train_nodes_from_test:
                        auxiliary_score = auxiliary_full[test_idx]
                    else:
                        auxiliary_score = auxiliary_full
                    print(
                        "[eval-cadets-context] cache_hit={} train_execs={} unseen_execs={} "
                        "mapped_nodes={} active_eval_nodes={} top_unseen={}".format(
                            auxiliary_details.get("cache_hit"),
                            auxiliary_details.get("train_exec_count"),
                            auxiliary_details.get("unseen_exec_count"),
                            auxiliary_details.get("mapped_node_count"),
                            int(np.sum(auxiliary_score > 0)),
                            auxiliary_details.get("top_unseen_execs"),
                        )
                    )
            print(
                f"[eval-knn] x_train={x_train.shape} x_test={result_x_test.shape} "
                f"y_test={result_y_test.shape}"
            )
            search_k = _parse_csv_int_list(main_args.entity_knn_search_k)
            search_metric = _parse_csv_str_list(main_args.entity_knn_search_metric)
            if len(search_k) == 0 and len(search_metric) == 0:
                eval_k = main_args.entity_knn_k if main_args.entity_knn_k > 0 else None
                eval_metric = main_args.entity_knn_metric
                test_auc, test_std, _, _ = evaluate_entity_level_using_knn(
                    dataset_name, x_train, result_x_test, result_y_test,
                    n_neighbors=eval_k, metric=eval_metric,
                    threshold_mode=main_args.entity_threshold_mode,
                    target_recall=main_args.entity_target_recall,
                    use_cache=main_args.entity_use_cache,
                    auxiliary_score=auxiliary_score,
                    auxiliary_weight=main_args.cadets_exec_context_weight,
                    auxiliary_name="cadets_unseen_exec" if auxiliary_score is not None else None
                )
            else:
                if len(search_k) == 0:
                    if main_args.entity_knn_k > 0:
                        search_k = [main_args.entity_knn_k]
                    else:
                        search_k = [10]
                if len(search_metric) == 0:
                    search_metric = [main_args.entity_knn_metric]

                print(f"[eval-knn-search] k_grid={search_k} metric_grid={search_metric}")
                best_metrics = None
                for k in search_k:
                    for metric in search_metric:
                        print(f"[eval-knn-search] evaluating k={k} metric={metric}")
                        auc_val, _, _, metrics = evaluate_entity_level_using_knn(
                            dataset_name, x_train, result_x_test, result_y_test,
                            n_neighbors=k, metric=metric,
                            threshold_mode=main_args.entity_threshold_mode,
                            target_recall=main_args.entity_target_recall,
                            use_cache=main_args.entity_use_cache,
                            auxiliary_score=auxiliary_score,
                            auxiliary_weight=main_args.cadets_exec_context_weight,
                            auxiliary_name="cadets_unseen_exec" if auxiliary_score is not None else None
                        )
                        if metrics is None:
                            continue
                        better = False
                        if best_metrics is None:
                            better = True
                        else:
                            if metrics["f1"] > best_metrics["f1"] + 1e-12:
                                better = True
                            elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and \
                                    metrics["precision"] > best_metrics["precision"] + 1e-12:
                                better = True
                            elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and \
                                    abs(metrics["precision"] - best_metrics["precision"]) <= 1e-12 and \
                                    metrics["recall"] > best_metrics["recall"] + 1e-12:
                                better = True
                        if better:
                            best_metrics = dict(metrics)
                            best_metrics["auc"] = float(auc_val)

                if best_metrics is None:
                    raise RuntimeError("entity-level KNN grid search failed to produce valid metrics")
                print(
                    "[eval-knn-search] best: k={} metric={} threshold_mode={} "
                    "AUC={:.4f} F1={:.4f} P={:.4f} R={:.4f} thr={:.6f}".format(
                        best_metrics["n_neighbors"],
                        best_metrics["metric"],
                        best_metrics["threshold_mode"],
                        best_metrics["auc"],
                        best_metrics["f1"],
                        best_metrics["precision"],
                        best_metrics["recall"],
                        best_metrics["threshold"],
                    )
                )
                test_auc, test_std = best_metrics["auc"], 0.0
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
