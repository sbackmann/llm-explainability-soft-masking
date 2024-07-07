import torch
import os
import dill
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from load_data import generics_kb, generics_kb_df
import numpy as np
import pandas as pd
import pickle as pkl
from embeddings import Embeddings
from utils import arg_parse
from _searcher import SimilaritySearch
from _sampler import Sampler
from scores import get_original_preds_probs, get_scores_replacement
from syntax_shap.syntaxshap import models
from syntax_shap.syntaxshap.utils.transformers import parse_prefix_suffix_for_tokenizer
from syntax_shap.syntaxshap.utils._filter_data import filter_data
from syntax_shap.syntaxshap.utils._exceptions import InvalidAlgorithmError
from syntax_shap.syntaxshap.explainers.other import LimeTextGeneration, Random, SVSampling, Ablation, HEDGEOrig
import syntax_shap.syntaxshap.explainers as explainers
from syntax_shap.syntaxshap.metrics import get_scores, save_scores
from syntax_shap.syntaxshap.utils._general import convert_to_token_expl


def main(args):

    ## Instantiate model
    if torch.cuda.is_available():
        device = "cuda"
        set_seed(args.seed) # should make no difference for embedding creation
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    else:
        print("Cuda not available")
        device = "cpu"

    # Initialize TextGeneration model
    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(lmmodel.tokenizer)
    keep_prefix = parsed_tokenizer_dict['keep_prefix'] # check that keep_prefix is not None, value 0 or 1
    keep_suffix = parsed_tokenizer_dict['keep_suffix'] # check that keep_prefix is not None, value 0 or 1


    ## Load data
    embedding_data, _ = generics_kb(args.data_dir, with_labels=True)

    filtered_data, filtered_ids = filter_data(embedding_data, tokenizer, args, keep_prefix, keep_suffix)
    if eval(args.shuffle):
        permutation_indices = np.random.permutation(len(filtered_data))
    else:
        permutation_indices = np.arange(len(filtered_data))

    data = filtered_data[permutation_indices] # check that permutation indices are in the range of 0 and len(data)
    data_ids = filtered_ids[permutation_indices]


    ## Retrieve word embeddings
    embedding_model = Embeddings(model, tokenizer, device, args)
    embeddings = embedding_model(embedding_data)

    if args.num_batch is not None:
        assert args.num_batch * args.batch_size < len(data), "Batch number is too large!"
        n_min = args.batch_size * args.num_batch
        n_max = args.batch_size * (args.num_batch + 1) if args.num_batch < len(data) // args.batch_size else len(data)
        print(f"Batch number {args.num_batch} of size {args.batch_size} is being used.")
        data = data[n_min:n_max]
        data_ids = data_ids[n_min:n_max]
    else:
        print(f"Batch number is not specified. Using all {len(data)} examples.")
    print("Length of data", len(data))
    
    #### Check if the explanations exist ####
    explanations_save_dir = os.path.join(args.result_save_dir, f'explanations/test/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}')
    os.makedirs(explanations_save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}.pkl"
    if os.path.exists(os.path.join(explanations_save_dir, filename)):
        print("Loading explanations...")
        with open(os.path.join(explanations_save_dir, filename), "rb") as f:
            results = pkl.load(f)
    else:
        #### Explain the model ####
        # Choose appropriate explainer based on specified algorithm
        if args.algorithm == "random":
            explainer = Random(lmmodel, lmmodel.tokenizer)
        elif args.algorithm == "partition":
            explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
        elif args.algorithm == "hedge":
            explainer = explainers.HEDGE(lmmodel, lmmodel.tokenizer, model)
        elif args.algorithm == "hedge_orig":
            explainer = HEDGEOrig(lmmodel, lmmodel.tokenizer)
        elif args.algorithm == "lime":
            explainer_save_dir = os.path.join(args.result_save_dir, f"explainer/seed_{args.seed}")
            os.makedirs(explainer_save_dir, exist_ok=True)
            explainer = LimeTextGeneration(lmmodel, filtered_data[:1000])
            #if os.path.exists(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl")):
                #print("Loading LIME explainer...")
                #explainer = dill.load(open(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl"), "rb"))
            #else:
                #explainer = LimeTextGeneration(lmmodel, filtered_data[:1000])
                #with open(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl"), "wb") as file:
                    #dill.dump(explainer, file)
        elif args.algorithm == "shap":
            explainer = Random(lmmodel, lmmodel.tokenizer)
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="shap")
        elif args.algorithm == "syntax":
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="syntax")
        elif args.algorithm == "syntax-w":
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="syntax-w")
        elif args.algorithm == "svsampling":
            explainer = SVSampling(lmmodel, lmmodel.tokenizer, model)
        elif args.algorithm == "ablation":
            explainer = Ablation(lmmodel, lmmodel.tokenizer, model)
        else:
            raise InvalidAlgorithmError("Unknown algorithm type passed: %s!" % args.algorithm)

        explanations = explainer(data)

        #### Save the shap values ####
        if args.algorithm == "lime":
            explanations = explainer._s
        else: 
            explanations = explanations.values

        results = []
        for i in range(len(explanations)):
            token_ids = lmmodel.tokenizer.encode(data[i])
            tokens = [lmmodel.tokenizer.decode(token_id) for token_id in token_ids]
            if args.algorithm == "lime":
                print('data[i]', data[i])
                print('explanations[i]', explanations[i])
                token_explanation = convert_to_token_expl(data[i], explanations[i], lmmodel.tokenizer, keep_prefix=keep_prefix)
                print("length of token_ids", len(token_ids))
                print("length of token_explanation", len(token_explanation))
                print("token_explanation", token_explanation)
            else:
                token_explanation = explanations[i]
            assert len(token_explanation) + keep_prefix == len(token_ids), "Length of explanations and data do not match!"
            results.append({'input_id': data_ids[i], 'input': data[i], 'tokens': tokens, 'token_ids': token_ids, 'explanation': token_explanation})
        with open(os.path.join(explanations_save_dir, filename), "wb") as f:
            pkl.dump(results, f)

    print("Done!")
    
    #### Evaluate the explanations ####
    scores_save_dir = os.path.join(args.result_save_dir, f'scores/test/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}')
    os.makedirs(scores_save_dir, exist_ok=True)
    filename = f"scores_{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}_{args.threshold}.pkl"
    if os.path.exists(os.path.join(scores_save_dir, filename)):
        with open(os.path.join(scores_save_dir, filename), "rb") as f:
            scores = pkl.load(f)
    else:
        # Calculate scores for explanations
        step_size = 50
        #for i in range(0, len(results), step_size):
        #    results_df = pd.DataFrame(results[i:min(len(results), i+step_size)])
        for i in range(10):
            threshold = i/10
            results_df = pd.DataFrame(results)
            print(f"Calculating scores for {len(results_df['input_id'])} explained instances...")
            scores = get_scores(results_df, lmmodel, threshold, args)
            print("scores", scores)
            save_scores(args, scores, threshold)

    ## Compute distances and ranks in embedding space
    searcher = SimilaritySearch(tokenizer, args.padding_length, args.seed, device)
    distances, ranks = searcher(embeddings, filtered_ids, embedding_data, args)

    results_df = pd.DataFrame(results)

    preds_probs = get_original_preds_probs(results_df, lmmodel, args.threshold)

    ## Compute scores for resampled tokens
    for i in range(10):
        threshold = i/10
        print(f"Calculating scores for {len(results_df['input_id'])} sampled instances with sampling method: {args.sampling_method}...")
        sample_scores, samples = get_scores_replacement(results_df, ranks, embedding_data, filtered_ids, lmmodel, threshold, args, preds_probs)
        j = 3700
        print(results_df["input"][j])
        #print(samples[j])
        print(results_df["explanation"][j])
        #sample_scores = get_scores_replacement(samples_df, lmmodel, threshold, args.algorithm, args, preds_probs)
        #print("scores", sample_scores)
        save_scores(args, sample_scores, threshold)



if __name__ == "__main__":
    parser, args = arg_parse()
    main(args)