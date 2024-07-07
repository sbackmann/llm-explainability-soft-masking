import numpy as np

def get_embedding_mask(tokenizer, embedding_data, padding_length, filtered_ids):
    tokens = tokenizer(embedding_data.tolist(), return_tensors="pt", padding="max_length", max_length=padding_length, add_special_tokens=True)
    mask = (tokens["input_ids"] != 0) & (tokens["input_ids"] != 2)
    mask_filtered = np.zeros((len(embedding_data), padding_length), dtype=bool)
    mask_filtered[filtered_ids] = np.ones(padding_length, dtype=bool)
    query_mask = mask & mask_filtered
    return mask.numpy(), query_mask.numpy()


def unpack_explanations(explanations):
    explanations_unpacked = [x for xs in explanations for x in xs]
    return explanations_unpacked
def normalize_mask(mask):
    return (mask - min(mask)) / (max(mask) - min(mask))

    