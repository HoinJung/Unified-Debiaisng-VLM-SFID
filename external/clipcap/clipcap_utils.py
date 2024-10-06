
import os
import pandas as pd
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import torch.nn.functional as nnf
import numpy as np
import pandas as pd 
import torch




masculine = ['man','men','male','father','gentleman','gentlemen','boy','boys','uncle','husband','actor',
            'prince','waiter','son','he','his','him','himself','brother','brothers', 'guy', 'guys',
            'emperor','emperors','dude','dudes','cowboy','boyfriend','chairman','policeman','policemen']
feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
            'princess','waitress','daughter','she','her','hers','herself','sister','sisters', 'queen',
            'queens','pregnant','girlfriend','chairwoman','policewoman','policewomen']
gender_words = masculine + feminine

neutral = ['person','people','parent','parents','child','children','spouse','server','they','their','them',
           'theirs','baby','babies','partner','partners','friend','friends','spouse','spouses','sibling',
           'siblings', 'chairperson','officer', 'surfer', 'kid', 'kids']

# function to decide gender
def decide_gender(sent_tokens):
    gender_list = []
    for token in sent_tokens:
        token = token.lower()
        if token in masculine:
            gender_list.append('Male')
        if token in feminine:
            gender_list.append('Female')
        if token in neutral:
            gender_list.append('Neut')

    if 'Male' in gender_list and 'Female' not in gender_list:
        gender = 'Male'
    elif 'Male' not in gender_list and 'Female' in gender_list:
        gender = 'Female'
    elif 'Male' in gender_list and 'Female' in gender_list:
        gender = 'Both'
    elif 'Neut' in gender_list:
        gender = 'Neut'
    else:
        gender = 'None'

    return gender


# Function to load existing results or create a new list
def load_results(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename).to_dict('records')
    else:
        return []

# Function to save results to a CSV file
def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

# Check if a result for a specific image_id already exists
def is_result_exist(results, image_id):
    return any(result['image_id'] == image_id for result in results)

prefix_length=10
    
def generate(
        model,
        tokenizer,
        text_important_indices=None,text_mean_features_lowconfidence=None,
        tokens=None,
        prompt=None,
        embed=None,
        mode=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    embed = model.clip_project(embed).reshape(1, prefix_length, -1)
    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)
                
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated,text_important_indices=text_important_indices,text_mean_features_lowconfidence=text_mean_features_lowconfidence,mode=mode)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
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

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)
    
    return generated_list[0]