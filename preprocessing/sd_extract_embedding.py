import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import sys
sys.path.append("./")
from external.SD.sdxl_code import CustomStableDiffusionPipeline,CustomUNet2DConditionModel
from diffusers import StableDiffusionPipeline
import torch

dataset = load_dataset("LabHC/bias_in_bios")
train_dataset = dataset['train']
val_dataset = dataset['dev']
test_dataset = dataset['test']


model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "CompVis/stable-diffusion-v1-4"
pipe1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = CustomStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

custom_unet = CustomUNet2DConditionModel(
    # Ensure you pass all necessary arguments that the original UNet2DConditionModel requires
    in_channels=pipe1.unet.config.in_channels,
    out_channels=pipe1.unet.config.out_channels,
    layers_per_block=pipe1.unet.config.layers_per_block,
    block_out_channels=pipe1.unet.config.block_out_channels,
    down_block_types=pipe1.unet.config.down_block_types,
    up_block_types=pipe1.unet.config.up_block_types,
    attention_head_dim=pipe1.unet.config.attention_head_dim,
    cross_attention_dim=pipe1.unet.config.cross_attention_dim,
    use_linear_projection=pipe1.unet.config.use_linear_projection,
    sample_size=pipe1.unet.config.sample_size 
).to(torch.float16).to(device) 

custom_unet.load_state_dict(pipe1.unet.state_dict())
pipe.unet = custom_unet.to(torch.float16).to(device)
pipe = pipe.to(torch.float16).to(device)


# Define a function to process and save the dataset
def process_and_save_text_embedding(dataset, split_name, batch_size=10, max_samples=20000):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder_embeddings = []
    decoder_down_embeddings = []
    decoder_mid_embeddings = []
    professions = []
    genders = []
    total_samples = 0
    
    for batch in tqdm(data_loader, desc=f"Processing {split_name} set"):
        if total_samples >= max_samples:
            break
        hard_texts = batch['hard_text']
        professions_batch = batch['profession']
        genders_batch = batch['gender']
        
        # Assuming encoder.forward() takes a list of texts and returns embeddings
        with torch.no_grad():
            output = pipe(hard_texts)  
            encoder_embedding = output[1]  # shape [20,77,768]
            down_embedding = output[2]      # shape [20,1280,8,8]
            mid_embedding = output[3]       # shape [20,1280,8,8]
            
            #### 
            ## TODO: Reshape and reduce the embeddings
            ####
            
            # Reshape and reduce encoder embedding: [20,77,768] -> [10,768]
            encoder_embedding = encoder_embedding.view(batch_size, 2, 77, 768)
            encoder_embedding = encoder_embedding.mean(dim=[1, 2])  # Average over the 2 duplicates and 77 tokens

            # Reshape and reduce down embedding: [20,1280,8,8] -> [10,1280]
            down_embedding = down_embedding.view(batch_size, 2, 1280, 8, 8)
            down_embedding = down_embedding.mean(dim=[1, 3, 4])  # Average over the 2 duplicates and spatial dimensions

            # Reshape and reduce mid embedding: [20,1280,8,8] -> [10,1280]
            mid_embedding = mid_embedding.view(batch_size, 2, 1280, 8, 8)
            mid_embedding = mid_embedding.mean(dim=[1, 3, 4])  # Average over the 2 duplicates and spatial dimensions

            # Append the embeddings to their respective lists
            encoder_embeddings.append(encoder_embedding)
            decoder_down_embeddings.append(down_embedding)
            decoder_mid_embeddings.append(mid_embedding)

        professions.extend(professions_batch)
        genders.extend(genders_batch)
        total_samples += batch_size

    #### 
    ## TODO: Stack the lists to form a single tensor for each embedding type
    ####
    
    # Stack all batches into a single tensor
    encoder_embeddings = torch.vstack(encoder_embeddings)  # Shape [N, 768]
    decoder_down_embeddings = torch.vstack(decoder_down_embeddings)  # Shape [N, 1280]
    decoder_mid_embeddings = torch.vstack(decoder_mid_embeddings)  # Shape [N, 1280]
    
    professions = torch.tensor(professions)
    genders = torch.tensor(genders)
    
    # Save as .pt files
    torch.save({
        'encoder_embedding': encoder_embeddings,
        'profession': professions,
        'gender': genders
    }, f"embedding/sd_bios_encoder_{split_name}_dataset.pt")
    
    torch.save({
        'decoder_mid_embedding': decoder_mid_embeddings,
        'decoder_down_embedding': decoder_down_embeddings,
        'profession': professions,
        'gender': genders
    }, f"embedding/sd_bios_decoder_{split_name}_dataset.pt")

# Process and save the datasets
process_and_save_text_embedding(train_dataset, 'train', max_samples=20000)
process_and_save_text_embedding(val_dataset, 'val', max_samples=10000)