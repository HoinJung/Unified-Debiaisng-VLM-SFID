import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

import torch
import torch.nn.functional as nnf
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm
import random
import numpy as np
import sys
sys.path.append('./')
from external.clipcap import clipcap_model

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

print(f"Extract CLIP-CAP's decoder embedding.")

clip_model = "ViT-B/32"
clip_name = clip_model.replace("/", "").replace('-', '')
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index

embedding = torch.load(f'embedding/fairface_{clip_name}_train.pt')
embedding_val = torch.load(f'embedding/fairface_{clip_name}_val.pt')

X_train = embedding['image_embeddings']
y_train = embedding['sensitive_attributes'][:, 1]
X_test = embedding_val['image_embeddings']
y_test = embedding_val['sensitive_attributes'][:, 1]
def process_embeddings(X, y, model, prefix_length, tokenizer, file_name):
    if os.path.exists(file_name):
        print(f"File already exists at: {file_name}. Skipping.")
        return

    total_embeddings = []
    gender_list = []
    entry_count = 1
    entry_length = 67
    top_p = 0.8
    temperature = 1.0
    stop_token_index = tokenizer.encode('.')[0]
    filter_value = -float("Inf")

    for i in tqdm(range(len(X))):
        
        embed = X[i].to(device).float()
        gender = y[i]
        embed = model.clip_project(embed).reshape(1, prefix_length, -1)
        tokens = None
        embeds=[]
        with torch.no_grad():
            generated = embed
            for _ in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                feature = model.gpt.interim_hidden_state
                # feature = model.transformer.interim_hidden_state

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break
                embeds.append(torch.mean(feature.squeeze(0), dim=0, keepdim=True))
            # Take the mean of the sequence dimension (X) before appending
            features=  torch.mean(torch.concat(embeds),dim=0).unsqueeze(0)
            
            total_embeddings.append(features.cpu())
            gender_list.append(gender)

    # Concatenate all the mean-pooled features
    embeddings = torch.cat(total_embeddings, dim=0)  # Shape: (N, 768)
    genders = torch.tensor(gender_list)
    
    # Save the embeddings and gender labels
    torch.save({"hidden_states": embeddings, "sensitive_attributes": genders}, file_name)
    print(f"File saved at: {file_name}")
# Initialize and load the model
prefix_length = 10
model_path = 'external/clipcap/clip_cap_coco_weight.pt'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = clipcap_model.ClipCaptionModel(prefix_length, device=device)
model.load_state_dict(torch.load(model_path),strict=False)

model = model.eval().to(device)

# Process and save train embeddings
process_embeddings(X_train, y_train, model, prefix_length, tokenizer, f'embedding/clip_cap_decoder_fairface_train.pt')

# Process and save test embeddings
process_embeddings(X_test, y_test, model, prefix_length, tokenizer, f'embedding/clip_cap_decoder_fairface_test.pt')
