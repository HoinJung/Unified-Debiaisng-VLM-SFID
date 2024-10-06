import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import clip
def ImageLoader():
    root_dir = '../data/fairface/'
    train_data = pd.read_csv(os.path.join(root_dir, 'fairface_label_train.csv'))
    valid_data = pd.read_csv(os.path.join(root_dir, 'fairface_label_val.csv'))
    test_data = None
    gender_map = {'Female': 0, 'Male': 1}
    race_map = {'East Asian': 0, 'Indian': 1, 'Black': 2, 'White': 3, 'Middle Eastern': 4, 'Latino_Hispanic': 5, 'Southeast Asian': 6}
    age_map = {
        '0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3, '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 
        'more than 70': 8
    }
    
    # Apply mappings
    train_data['gender'] = train_data['gender'].map(gender_map)
    train_data['race'] = train_data['race'].map(race_map)
    train_data['age'] = train_data['age'].map(age_map)

    valid_data['gender'] = valid_data['gender'].map(gender_map)
    valid_data['race'] = valid_data['race'].map(race_map)
    valid_data['age'] = valid_data['age'].map(age_map)

    return train_data.reset_index(drop=True),  \
           valid_data.reset_index(drop=True) if valid_data is not None else None, \
           test_data.reset_index(drop=True) if test_data is not None else None

class ImageDataset(Dataset):
    def __init__(self, data, sens_idx, label_idx, root_dir=None, transform=None):
        self.transform = transform
        self.data = data
        self.root_dir = root_dir
        self.sens_idx = sens_idx
        self.label_idx = label_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        if isinstance(self.sens_idx, list):
            sens = self.data.loc[idx, self.sens_idx].values.astype(int)
        else:
            sens = self.data[self.sens_idx][idx]
            sens = max(int(sens), 0)
        if isinstance(self.label_idx, list):
            label = (self.data.loc[idx, self.label_idx].values > 0).astype(int)
        elif self.label_idx is None:
            label = None            
        else:
            label = self.data[self.label_idx][idx]
            label = max(int(label), 0)
        if label is None:
            return image, sens
        elif sens is None:
            return image, label
        else:
            return image, sens, label

label = None
sens = ["age", "gender", "race"]
root_dir = '../data/fairface'

train_dataset, val_dataset, _ = ImageLoader()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_list = ['RN50', 'ViT-B/32']  # Choose the baseline in CLIP
print(f"Extract CLIP image and text embedding for debiasing with baseline models {model_list}.")

for clip_model in model_list:
    print(f"Processing model {clip_model}.")
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    prefix = clip_model.replace("/", "").replace('-', '')

    # Process training set
    save_path_train = f'embedding/fairface_{prefix}_train.pt'
    if os.path.exists(save_path_train):
        print(f"Training file already exists at: {save_path_train}. Skipping.")
    else:
        bs = 128
        train_data = ImageDataset(train_dataset, sens, label, root_dir, preprocess)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, num_workers=32)

        with torch.no_grad():
            image_embeddings_list = []
            sensitive_attributes_list = []

            for x_batch, s_batch in tqdm(train_loader):
                x_batch = x_batch.to(device)
                image_features = model.encode_image(x_batch)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                image_embeddings_list.append(image_features.cpu())
                sensitive_attributes_list.append(s_batch)

            torch.save({
                'image_embeddings': torch.cat(image_embeddings_list),
                'sensitive_attributes': torch.cat(sensitive_attributes_list),
            }, save_path_train)
            print(f'Training file saved at: {save_path_train}')

    # Process validation set
    save_path_val = f'embedding/fairface_{prefix}_val.pt'
    if os.path.exists(save_path_val):
        print(f"Validation file already exists at: {save_path_val}. Skipping.")
    else:
        bs = 128
        val_data = ImageDataset(val_dataset, sens, label, root_dir, preprocess)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=32)

        with torch.no_grad():
            image_embeddings_list = []
            sensitive_attributes_list = []

            for x_batch, s_batch in tqdm(val_loader):
                x_batch = x_batch.to(device)
                image_features = model.encode_image(x_batch)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                image_embeddings_list.append(image_features.cpu())
                sensitive_attributes_list.append(s_batch)

            torch.save({
                'image_embeddings': torch.cat(image_embeddings_list),
                'sensitive_attributes': torch.cat(sensitive_attributes_list),
            }, save_path_val)
            print(f'Validation file saved at: {save_path_val}')

import os
from datasets import load_dataset

train_dataset = load_dataset("LabHC/bias_in_bios", split='train')
test_dataset = load_dataset("LabHC/bias_in_bios", split='test')
val_dataset = load_dataset("LabHC/bias_in_bios", split='dev')

model_list = ['RN50', 'ViT-B/32']  # Choose the baseline in CLIP
for clip_model in model_list:
    print(f"Text for model {clip_model}.")
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    for split in ['train', 'test', 'val']:
        if split == 'train':
            dataset = train_dataset
        elif split == 'val':
            dataset = val_dataset
        elif split == 'test':
            dataset = test_dataset

        batch_size = 128  # You can adjust the batch size according to your GPU memory
        prefix = clip_model.replace("/", "").replace('-', '')
        save_path = f'embedding/{prefix}_bios_bias_text_{split}.pt'
        
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"File already exists at: {save_path}. Skipping.")
            continue
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        max_tokens = 77
        # Inference loop
        all_text_features = []
        all_genders = []
        all_profession = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                texts = batch['hard_text']  # Adjust this depending on your dataset structure
                gender = batch['gender']
                profession = batch['profession']
                text_features = clip.tokenize(texts, truncate=True).to(device)
                text_features = model.encode_text(text_features)
                all_text_features.append(text_features)
                all_genders.append(gender)
                all_profession.append(profession)

        all_text_features = torch.cat(all_text_features, dim=0)
        all_genders = torch.cat(all_genders, dim=0)
        all_profession = torch.cat(all_profession, dim=0)

        # Save the results if the file doesn't exist
        torch.save({
            'text_embeddings': all_text_features.cpu(),
            'sensitive_attributes': all_genders.cpu(),
            'all_profession': all_profession.cpu()
        }, save_path)
        print(f'File saved at: {save_path}')
