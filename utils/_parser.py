import argparse

MODEL_DIR = "/cluster/courses/xaiml/data/sbackmann-sem-project/models/gemma-2b/"
HIDDEN_STATES_DIR = "/home/sbackmann/sem-project/results/embeddings/test"
DATA_DIR = "/home/sbackmann/sem-project/"
RESULT_DIR = DATA_DIR + "results/"

def arg_parse():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", help="random seed", type=int, default=42
    )

    parser.add_argument(
        "--shuffle", type=str, default='False', help="shuffle the data if 'True' else 'False'", choices=["True", "False"]
    )

    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR
    )

    parser.add_argument(
        "--model_dir", type=str, default=MODEL_DIR
    )

    parser.add_argument(
        "--hidden_states_dir", type=str, default=HIDDEN_STATES_DIR
    )

    parser.add_argument(
        "--data_save_dir", type=str, default=DATA_DIR
    )

    parser.add_argument(
        "--result_save_dir", type=str, default=RESULT_DIR
    )
    parser.add_argument(
        "--dataset", type=str, default="generics"
    )

    parser.add_argument(
        "--model_name", type=str, default="gemma-2b"
    )

    parser.add_argument(
        "--batch_size", type=int, default=16
    )

    parser.add_argument(
        "--num_batch", type=int, default=None
    )

    parser.add_argument(
        # Set to #tokens (including <bos>) of longest input
        "--padding_length", type=int, default=24
    )

    parser.add_argument(
        "--embedding_layer", type=str, default="lower", choices=["upper", "lower"]
    )
    
    parser.add_argument(
        "--algorithm",
        help="The type of explanations algorithm",
        type=str,
        default="syntax", choices=["random", "lime", "partition", "shap", "syntax", "syntax-w", "svsampling", "ablation", "hedge_orig"]
    )

    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine", choices=["cosine", "euclidean"]
    )

    parser.add_argument(
        "--sampling_method",
        type=str,
        default="top10", choices=["complete", "top50", "top10", "top10_normal"]
    )

    parser.add_argument(
        "--explanations_save_dir",
        help="Directory where explanations are saved",
        type=str,
        default=DATA_DIR
    )

    parser.add_argument(
        # Currently not used, code always runs all thresholds
        "--threshold",
        help="The percentage of important indices ",
        type=float,
        default=0.3
    )

    args, unknown = parser.parse_known_args()
    return parser, args