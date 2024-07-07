import numpy as np
from syntax_shap.syntaxshap.metrics import generate_explanatory_masks, run_model, compute_acc_at_k, compute_prob_diff_at_k, replace_token_randomly
from tqdm import tqdm
import torch
import pickle as pkl
import os
from _sampler import Sampler

def get_original_preds_probs(
    results,
    pipeline,
    k: float,
    token_id: int = 0
):
    # Generate explanatory masks
    masks = generate_explanatory_masks(results["input"], results["explanation"], results["tokens"], k, pipeline.tokenizer, token_id)
    # Initialize lists to store predictions and probabilities
    preds_orig, probs_orig = [], []
    
    N = len(results["input"])
    print("Number of explained instances", N)
    
    # Iterate through all inputs
    for i, str_input in enumerate(tqdm(results["input"])):
        # Skip if mask is None
        if masks[i] is None:
            print("masks[i] is None for input", str_input, " - skipping...")
            N -= 1
            continue
        else:
            row_args = [str_input]
            mask = np.array(masks[i])

            # Get predictions and probabilities for original, keep, remove, and keep with random replacements
            orig = run_model(row_args, None, pipeline)
            preds_orig.append(orig[0])
            probs_orig.append(orig[1])
        
    print("Number of explained instances after removing None masks", N)

    # Concatenate predictions and probabilities lists

    preds_orig, probs_orig = np.concatenate(preds_orig).astype(int), np.concatenate(probs_orig)
    
    top_1_probs_orig = torch.Tensor([probs_orig[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()

    return preds_orig, top_1_probs_orig

def replace_tokens(
    inputs,
    masks: np.ndarray,
    tokenizer,
    args
):
    sampler = Sampler(args.sampling_method, tokenizer, args.padding_length)
    samples = sampler(ranks, inputs, embedding_data, filtered_ids)

    for i, sentence_tokenized in enumerate(inputs):
        mask_i = 1 - masks[i]
        sentence_np = np.array(sentence_tokenized)
        sentence_np[mask_i.astype(bool)] = np.random.choice(vocab, size=mask_i.sum().astype(int))
        inputs[i] = sentence_np.tolist()
    inputs_str = tokenizer.batch_decode(inputs)

    return inputs_str

def get_scores_replacement(
    results,
    ranks,
    embedding_data,
    filtered_ids,
    pipeline,
    k: float,
    args,
    preds_probs, 
    token_id: int = 0
) -> dict:
    """
    Calculates scores for the explanations.

    Args:
        str_inputs (List[str]): List of input strings.
        input_ids (List[int]): List of input IDs.
        shapley_scores: Shapley scores.
        pipeline: Pipeline object.
        k (float): The percentage of important indices.
        token_id (int, optional): Token ID. Defaults to 0.

    Returns:
        dict: Dictionary containing computed scores.
    """
    # Generate explanatory masks
    masks = generate_explanatory_masks(results["input"], results["explanation"], results["tokens"], k, pipeline.tokenizer, token_id)
    if args.sampling_method == "top10_normal":
        iters = 10
    else:
        iters = 1
    fid_keep_repls = []

    for it in range(iters):
        sampler = Sampler(args.sampling_method, pipeline.tokenizer, args.padding_length)
        samples, samples_list = sampler(ranks, results, embedding_data, filtered_ids, masks)

        # Initialize lists to store predictions and probabilities
        preds_keep_repl, probs_keep_repl = [], []
        # preds_keep = []

        # Initialize lists to store valid input ids and inputs
        valid_ids = []

        with open(os.path.join(args.result_save_dir, f"scores/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}/scores_{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}_{k}.pkl"), "rb") as f:
            scores_orig = pkl.load(f)
        
        #assert results["input_id"].tolist() == scores_orig["input_id"]
        N = len(results["input"])
        print("Number of explained instances", N)

        # Iterate through all inputs
        for i, str_input in enumerate(tqdm(samples)):
            # Skip if mask is None
            if masks[i] is None:
                print("masks[i] is None for input", str_input, " - skipping...")
                N -= 1
                continue
            else:
                row_args = [str_input]
                mask = np.array(masks[i])
                # Get predictions and probabilities for keep with replacements

                keep_repl = run_model(row_args, mask, pipeline)
                preds_keep_repl.append(keep_repl[0])
                probs_keep_repl.append(keep_repl[1])

                # keep = run_model([results["input"][i]], mask, pipeline)
                # preds_keep.append(keep[0])

                valid_ids.append(results["input_id"][i])

        print("Number of explained instances after removing None masks", N)

        preds_orig = preds_probs[0]
        top_1_probs_orig = preds_probs[1]
        assert len(preds_orig) == len(preds_keep_repl)
        assert valid_ids == scores_orig["input_id"]
        print(preds_keep_repl)

        # Concatenate predictions and probabilities lists
        preds_keep_repl, probs_keep_repl = np.concatenate(preds_keep_repl).astype(int), np.concatenate(probs_keep_repl)
        # preds_keep = np.concatenate(preds_keep).astype(int)

        # Calculate fidelity
        top_1_probs_keep_repl = torch.Tensor([probs_keep_repl[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()

        fid_keep_repl = top_1_probs_orig - top_1_probs_keep_repl
        fid_keep_repls.append(fid_keep_repl)

    fid_keep_repl = np.mean(fid_keep_repls, axis=0)
    # scores_orig["y_orig"] = pipeline.tokenizer.batch_decode(preds_orig)
    # scores_orig["y_keep"] = pipeline.tokenizer.batch_decode(preds_keep)
    # scores_orig[f"y_repl_{args.similarity_metric}_{args.embedding_layer}_{args.sampling_method}"] = pipeline.tokenizer.batch_decode(preds_keep_repl)
    # scores_orig[f"tokens_repl_{args.similarity_metric}_{args.embedding_layer}_{args.sampling_method}"] = samples_list
    scores_orig[f"fid_keep_repl_{args.similarity_metric}_{args.embedding_layer}_{args.sampling_method}"] = fid_keep_repl.tolist()

    return scores_orig, fid_keep_repls