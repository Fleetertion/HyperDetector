import argparse


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
    parser.add_argument("--no_bsa", action="store_false", dest="use_bsa",
                        help="disable Block Self-Attention in encoder")
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
    parser.add_argument("--entity_knn_metric", type=str, default="euclidean", choices=["euclidean", "cosine"],
                        help="entity-level KNN distance metric")
    parser.add_argument("--entity_threshold_mode", type=str, default="legacy",
                        choices=["legacy", "f1", "target_recall"],
                        help="threshold strategy for entity-level KNN evaluation")
    parser.add_argument("--entity_target_recall", type=float, default=None,
                        help="target recall when --entity_threshold_mode target_recall")
    parser.add_argument("--entity_knn_search_k", type=str, default="",
                        help="comma-separated k grid for entity-level KNN search, e.g. 5,10,20")
    parser.add_argument("--entity_knn_search_metric", type=str, default="",
                        help="comma-separated metric grid for entity-level KNN search, e.g. euclidean,cosine")
    args = parser.parse_args()
    return args
