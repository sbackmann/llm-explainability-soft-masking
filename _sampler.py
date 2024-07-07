import numpy as np
from utils._general import get_embedding_mask, unpack_explanations, normalize_mask

class Sampler:

    def __init__(self, sampling_method, tokenizer, padding_length):
        self.sampling_method = sampling_method
        self.tokenizer = tokenizer
        self.padding_length = padding_length


    def __call__(self, ranks, results, embedding_data, filtered_ids, explanation_masks):
        embedding_mask, filtered_mask = get_embedding_mask(self.tokenizer, embedding_data, self.padding_length, filtered_ids)

        tokens = self.tokenizer(embedding_data.tolist(), return_tensors="pt", padding="max_length", max_length=self.padding_length, add_special_tokens=True)
        tokens_flattened = tokens["input_ids"][embedding_mask]
        print(tokens_flattened)
        index = 0
        resampled_inputs = []
        resampled_inputs_list = []
        for idx, result in results.iterrows():
            len_token = 0
            explanation = result["explanation"]
            explanation = unpack_explanations(explanation)
            i = 0

            while len_token != len(explanation):
                #print(result)
                explanation_normalized = normalize_mask(explanation)
                explanation_mask_i = 1 - explanation_masks[idx]
                explanation_normalized[explanation_mask_i.astype(bool)] = 1
                if self.sampling_method == "complete":
                    rank_indices = self.proportional_sampling(ranks, explanation_normalized, 1)
                elif self.sampling_method == "top50":
                    rank_indices = self.proportional_sampling(ranks, explanation_normalized, 2)
                elif self.sampling_method == "top10":
                    rank_indices = self.proportional_sampling(ranks, explanation_normalized, 10)
                elif self.sampling_method == "top10_normal":
                    # std is kind of arbitrary right now
                    rank_indices = self.normal_sampling(ranks, explanation_normalized, similarity=10, std=10)
                
                if i != 0:
                    print(i)
                    ## resample tokens in near vicinity if length doesn't match
                    rank_indices[explanation_normalized != 1] += int(np.ceil(abs(i)/2)  * ((-1) ** i))
                    explanation_normalized = np.clip(rank_indices, 0, ranks.shape[-1])
                resampled_token_indices = ranks[np.arange(index, index + len(explanation_normalized)), rank_indices]
                resampled_token_ids = tokens_flattened[resampled_token_indices]
                resampled_tokens = self.tokenizer.decode(resampled_token_ids)
                resampled_tokens_list = self.tokenizer.batch_decode(resampled_token_ids)
                len_token = len(self.tokenizer(resampled_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"][0])

                i += 1
                # Can't find tokens that result in same length (could potentially be increased)
                if i >= 1000:
                    raise ValueError("Max i reached")
            # if idx == 0:
            #     print(resampled_token_indices)
            #    print(resampled_tokens)
            resampled_inputs.append(resampled_tokens)
            resampled_inputs_list.append(resampled_tokens_list)
            index += len(explanation)
        return resampled_inputs, resampled_inputs_list
        

    def normal_sampling(self, ranks, explanation_normalized, similarity, std):
        rank_indices = np.fix(std * np.random.randn(len(explanation_normalized)) + (1 - explanation_normalized) * ranks.shape[-1]/similarity)
        rank_indices[explanation_normalized == 1] = 0
        rank_indices = np.clip(rank_indices, 0, ranks.shape[-1]/similarity)
        print(f"Ranks: {(1 - explanation_normalized) * ranks.shape[-1]/similarity}\n Samples: {rank_indices}\n")
        return rank_indices.astype(int)

    def proportional_sampling(self, ranks, explanation_normalized, similarity):
        return np.fix((1 - explanation_normalized) * ranks.shape[-1]/similarity - 1e-5).astype(int)



