import torch
import pickle as pkl
import os


class Embeddings:

    def __init__(self, model, tokenizer, device, args):

        self.model = model
        self.tokenizer = tokenizer
        self.padding_length = args.padding_length
        if args.model_name == "gemma-2b":
            self.hidden_size = 2048
        self.batch_size = args.batch_size
        self.hidden_states_dir = args.hidden_states_dir
        self.layer = args.embedding_layer
        self.device = device


    def dump_hidden_states(self, hidden_states):
        with open(os.path.join(self.hidden_states_dir, f"hidden_states_{self.layer}_tokenized.pkl"), "wb") as f:
            pkl.dump(hidden_states, f)


    def __call__(self, data):
        assert self.layer in ["upper", "lower"], "`layer` must be either 'upper' or 'lower'"
        hidden_path = os.path.join(self.hidden_states_dir, f"hidden_states_{self.layer}_tokenized.pkl")
        if os.path.exists(hidden_path):
            print("Loading pickled hidden_layer...")
            with open(hidden_path, "rb") as f:
                hidden_states = pkl.load(f)
        else:
            os.makedirs(self.hidden_states_dir)
            if self.layer == "upper":
                layer_id = -2
            elif self.layer == "lower":
                layer_id = 1
            # create empty tensor to store hidden states
            hidden_states = torch.zeros((len(data), self.padding_length, self.hidden_size))
            with torch.inference_mode():
                for i in range(0, len(data), self.batch_size):
                    inputs = self.tokenizer(data.tolist()[i:min(i+self.batch_size, len(data))], return_tensors="pt", padding="max_length", max_length=self.padding_length, add_special_tokens=True).to(self.device)
                    #print(inputs)
                    outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
                    hidden_states[i:min(i+self.batch_size, len(data))] = outputs.hidden_states[layer_id]
                    if i % (10 * self.batch_size) == 0:
                        print(i)
                        self.dump_hidden_states(hidden_states[:min(i+self.batch_size, len(data))])
            #print(hidden_states_upper)
            self.dump_hidden_states(hidden_states)
        return hidden_states