import argparse
import sys


def build_args():
    parser = argparse.ArgumentParser(description="MAGIC")
    parser.add_argument("--dataset", type=str, default="wget")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_fn", type=str, default='sce')
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--num_hidden", type=int, default=None,
                        help="override hidden embedding dimension; must be divisible by 4")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="override encoder layer count")
    parser.add_argument("--dropout", type=float, default=None,
                        help="override dropout for pipelines that expose model dropout")
    parser.add_argument("--hyper_k", type=int, default=4,
                        help="number of neighbors per center node when building k-hop hyperedges")
    parser.add_argument("--hyper_shuffle", action="store_true",
                        help="randomly sample k neighbors for each hyperedge")
    parser.add_argument("--bsa_heads", type=int, default=4,
                        help="number of attention heads in Block Self-Attention")
    parser.add_argument("--bsa_block_size", type=int, default=256,
                        help="chunk size for Block Self-Attention")
    parser.add_argument("--bsa_dropout", type=float, default=0.1,
                        help="dropout in Block Self-Attention")
    parser.add_argument("--bsa_gate_init", type=float, default=0.0,
                        help="initial residual gate value for Block Self-Attention")
    parser.add_argument("--bsa_scale", type=float, default=0.2,
                        help="maximum residual scale multiplier for Block Self-Attention")
    parser.add_argument("--use_bsa", action="store_true", default=False,
                        help="enable Block Self-Attention in encoder (default: enabled)")
    parser.add_argument("--threshold_mode", type=str, default="f1", choices=["f1", "target_recall"],
                        help="decision-threshold strategy for batch-level KNN evaluation")
    parser.add_argument("--target_recall", type=float, default=None,
                        help="target recall used when --threshold_mode target_recall")
    parser.add_argument("--max_epoch", type=int, default=None,
                        help="override training epochs for train.py")
    parser.add_argument("--log_interval", type=int, default=0,
                        help="log every N training graphs in entity-level training (0 disables per-step logs)")
    parser.add_argument("--entity_knn_k", type=int, default=0,
                        help="entity-level KNN neighbors; 0 means dataset default")
    parser.add_argument("--entity_knn_metric", type=str, default=None, choices=["euclidean", "cosine"],
                        help="entity-level KNN distance metric; unset means dataset default")
    parser.add_argument("--entity_use_cache", action="store_true", default=None,
                        help="reuse cached entity-level KNN distances when shape matches")
    parser.add_argument("--no_entity_use_cache", action="store_false", dest="entity_use_cache",
                        help="disable cached entity-level KNN distances")
    parser.add_argument("--entity_threshold_mode", type=str, default="legacy",
                        choices=["legacy", "f1", "target_recall"],
                        help="threshold strategy for entity-level KNN evaluation")
    parser.add_argument("--entity_target_recall", type=float, default=None,
                        help="target recall when --entity_threshold_mode target_recall")
    parser.add_argument("--entity_knn_search_k", type=str, default="",
                        help="comma-separated k grid for entity-level KNN search, e.g. 5,10,20")
    parser.add_argument("--entity_knn_search_metric", type=str, default="",
                        help="comma-separated metric grid for entity-level KNN search, e.g. euclidean,cosine")
    parser.add_argument("--cadets_exec_context", action="store_true", default=True,
                        help="enable CADETS unseen-exec context score during entity-level evaluation")
    parser.add_argument("--no_cadets_exec_context", action="store_false", dest="cadets_exec_context",
                        help="disable CADETS unseen-exec context score")
    parser.add_argument("--cadets_exec_context_weight", type=float, default=1.0,
                        help="weight for the CADETS unseen-exec context score")
    parser.add_argument("--apt_lr", type=float, default=None,
                        help="APT2021 override for train.lr")
    parser.add_argument("--apt_weight_decay", type=float, default=None,
                        help="APT2021 override for train.weight_decay")
    parser.add_argument("--apt_mask_rate", type=float, default=None,
                        help="APT2021 override for train.mask_rate")
    parser.add_argument("--apt_patience", type=int, default=None,
                        help="APT2021 override for train.early_stop_patience")
    parser.add_argument("--apt_min_delta", type=float, default=None,
                        help="APT2021 override for train.min_delta")
    parser.add_argument("--apt_warmup_epochs", type=int, default=None,
                        help="APT2021 override for train.warmup_epochs")
    parser.add_argument("--apt_hid_dim", type=int, default=None,
                        help="APT2021 override for model.hid_dim")
    parser.add_argument("--apt_num_layers", type=int, default=None,
                        help="APT2021 override for model.num_layers")
    parser.add_argument("--apt_hyper_k", type=int, default=None,
                        help="APT2021 override for model.hyper_k")
    parser.add_argument("--apt_dropout", type=float, default=None,
                        help="APT2021 override for model.dropout")
    parser.add_argument("--apt_bsa_heads", type=int, default=None,
                        help="APT2021 override for model.bsa_heads")
    parser.add_argument("--apt_bsa_block_size", type=int, default=None,
                        help="APT2021 override for model.bsa_block_size")
    parser.add_argument("--apt_temporal_block_size", type=int, default=None,
                        help="APT2021 override for model.temporal_block_size; <=0 means full temporal attention")
    parser.add_argument("--apt_use_bsa", dest="apt_use_bsa", action="store_true", default=None,
                        help="APT2021 enable model.use_bsa")
    parser.add_argument("--apt_no_bsa", dest="apt_use_bsa", action="store_false",
                        help="APT2021 disable model.use_bsa")
    parser.add_argument("--apt_use_temporal", dest="apt_use_temporal", action="store_true", default=None,
                        help="APT2021 enable model.use_temporal")
    parser.add_argument("--apt_no_temporal", dest="apt_use_temporal", action="store_false",
                        help="APT2021 disable model.use_temporal")
    parser.add_argument("--apt_use_norm", dest="apt_use_norm", action="store_true", default=None,
                        help="APT2021 enable model.use_norm")
    parser.add_argument("--apt_no_norm", dest="apt_use_norm", action="store_false",
                        help="APT2021 disable model.use_norm")
    parser.add_argument("--apt_kdt_k", type=int, default=None,
                        help="APT2021 override for eval.kdt_k")
    parser.add_argument("--apt_sweep_kdt_k", type=str, default=None,
                        help="APT2021 comma-separated eval.sweep_kdt_k")
    parser.add_argument("--apt_target_fpr", type=float, default=None,
                        help="APT2021 override for eval.default_target_fpr")
    parser.add_argument("--apt_sweep_target_fpr", type=str, default=None,
                        help="APT2021 comma-separated eval.sweep_target_fpr")
    parser.add_argument("--apt_recall_drop_budget", type=float, default=None,
                        help="APT2021 override for eval.recall_drop_budget")
    args = parser.parse_args()
    args._provided_flags = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args
