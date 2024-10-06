import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import sys
original_cwd = os.getcwd()
codi_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'codi'))
os.chdir(codi_dir)
sys.path.insert(0, codi_dir)
print("Current working directory:", os.getcwd())

from core.models.model_module_infer import model_module

dataset = load_dataset("LabHC/bias_in_bios")
train_dataset = dataset['train']
val_dataset = dataset['dev']
test_dataset = dataset['test']

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = model_module(data_dir='../../checkpoint/', pth=model_load_paths, fp16=True)  # turn on fp16=True if loading fp16 weights
inference_tester = inference_tester.cuda()
inference_tester = inference_tester.eval()
os.chdir(original_cwd)
def process_and_save_image_embedding(dataset, split_name, batch_size=40, max_samples=None):
    output_file = f"embedding/codi_bios_{split_name}_dataset_full.pt"
    
    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping processing.")
        return
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    text_embeddings = []
    down_image_embeddings = []
    mid_image_embeddings = []
    professions = []
    genders = []
    total_samples = 0
    
    for it, batch in enumerate(tqdm(data_loader, desc=f"Processing {split_name} set")):
        if total_samples >= max_samples:
            break
        hard_texts = batch['hard_text']
        professions_batch = batch['profession']
        genders_batch = batch['gender']
        
        with torch.no_grad():
            conditioning = []
            ctx = inference_tester.net.clip_encode_text(hard_texts)
            utx = inference_tester.net.clip_encode_text(len(hard_texts) * [""]).cuda()
            conditioning.append(torch.cat([utx, ctx]))
            shapes = []
            image_size = 256
            h, w = [image_size, image_size]
            shape = [len(hard_texts), 4, h // 8, w // 8]
            shapes.append(shape)
            z, _, image_rep = inference_tester.sampler.sample(
                steps=50,
                shape=shapes,
                condition=conditioning,
                unconditional_guidance_scale=7.5,
                xtype=['image'], 
                condition_types=['text'],
                eta=0.0,
                verbose=False,
                mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1}
            )
        
        down = image_rep[0]
        mid = image_rep[1]
        current_batch_size = int(down.shape[0] / 2)
        
        down = down.view(current_batch_size, 2, 1280, 4, 4)
        down_averaged_tensor = down.mean(dim=1)
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        down_pooled_tensor = avg_pool(down_averaged_tensor).squeeze(-1).squeeze(-1)

        mid = mid.view(current_batch_size, 2, 1280, 4, 4)
        mid_averaged_tensor = mid.mean(dim=1)
        mid_pooled_tensor = avg_pool(mid_averaged_tensor).squeeze(-1).squeeze(-1)

        ctx = ctx.squeeze(1).cpu()
        
        if it <= 3:
            print(ctx.shape)
            print(down_pooled_tensor.shape)
            print(mid_pooled_tensor.shape)
            
        text_embeddings.extend(ctx.numpy())
        down_image_embeddings.extend(down_pooled_tensor.cpu().numpy())
        mid_image_embeddings.extend(mid_pooled_tensor.cpu().numpy())
        professions.extend(professions_batch)
        genders.extend(genders_batch)
        total_samples += current_batch_size
        
    torch.save({
        'down_image_embedding': torch.tensor(down_image_embeddings),
        'mid_image_embedding': torch.tensor(mid_image_embeddings),
        'text_embedding': torch.tensor(text_embeddings),
        'profession': torch.tensor(professions),
        'gender': torch.tensor(genders)
    }, output_file)

# Process and save the datasets
process_and_save_image_embedding(train_dataset, 'train', max_samples=20000)
process_and_save_image_embedding(val_dataset, 'val', max_samples=10000)

