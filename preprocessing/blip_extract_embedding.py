import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)  # Maps GPU 2 to index 0
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import sys
sys.path.append('./')
sys.path.append('external/BLIP/')
from external.BLIP.models.blip import blip_decoder
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

model_path = 'external/BLIP/model_base_capfilt_large.pth'
image_size = 384
model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
root_dir = '../data/fairface/'
def ImageLoader():

    

    train_data = pd.read_csv(os.path.join(root_dir, 'fairface_label_train.csv'))
    valid_data = pd.read_csv(os.path.join(root_dir, 'fairface_label_val.csv'))
    test_data = None


    gender_map = {'Female': 0, 'Male': 1}
    race_map = {'East Asian': 0, 'Indian': 1, 'Black' : 2,'White': 3, 'Middle Eastern': 4, 'Latino_Hispanic': 5,'Southeast Asian':6}
    age_map = {
    '0-2':0, '3-9':1, '10-19':2, '20-29':3,  '30-39':4,  '40-49':5, '50-59':6, '60-69':7, 
    'more than 70':8}
    
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
    def __init__(self, data,  sens_idx, label_idx, root_dir = None, transform = None):
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
            # image = self.transform(image).unsqueeze(0).to(device)

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
            return  image, sens
        elif sens is None:
            return  image, label
        else:
            return  image, sens, label
            
label = None
sens = ["age", "gender", "race"]
train_dataset, val_dataset,_ = ImageLoader()


bs = 64
print(f"Extract BLIP's encoder embedding.")
def process_dataset(dataset, save_path):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping processing.")
        return
    
    dataloader = DataLoader(ImageDataset(dataset, sens, label, root_dir, transform), 
                            batch_size=bs, shuffle=False, num_workers=4)
    
    image_embeddings_list = []
    sensitive_attributes_list = []
    
    # Assuming model is properly defined and initialized
    with torch.no_grad():
        for x_batch, s_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            image_features = model.visual_encoder(x_batch)
            
            # Average over the sequence length dimension
            image_features = image_features.mean(dim=1)  # Now shape is [Batch_size, Dimension]
            
            # Accumulate results
            image_embeddings_list.append(image_features.cpu())
            sensitive_attributes_list.append(s_batch.cpu())
            
    # Concatenate all data once after the loop
    image_embeddings = torch.cat(image_embeddings_list)
    sensitive_attributes = torch.cat(sensitive_attributes_list)
    
    # Save the unified file
    torch.save({
        'image_embeddings': image_embeddings,
        'sensitive_attributes': sensitive_attributes,
    }, save_path)
    print(f"Saved embeddings to {save_path}")

# Process training set
process_dataset(train_dataset, 'embedding/fairface_BLIP_encoder_train.pt')

# Process validation set
process_dataset(val_dataset, 'embedding/fairface_BLIP_encoder_val.pt')



print(f"Extract BLIP's decoder embedding.")
def process_decoder_embeddings(dataset, save_path):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping processing.")
        return
    
    dataloader = DataLoader(ImageDataset(dataset, sens, label, root_dir, transform), 
                            batch_size=bs, shuffle=False, num_workers=4)

    image_embeddings_list = []
    sensitive_attributes_list = []
    
    with torch.no_grad():
        for x_batch, s_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            image_embeds = model.visual_encoder(x_batch)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
            prompt = ['a picture of '] * x_batch.size(0)
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            input_ids[:, 0] = model.tokenizer.bos_token_id
            input_ids = input_ids[:, :-1]

            outputs = model.text_decoder.generate(
                input_ids=input_ids, max_length=30, min_length=10, num_beams=1,
                eos_token_id=model.tokenizer.sep_token_id, pad_token_id=model.tokenizer.pad_token_id,
                repetition_penalty=1.0, **model_kwargs
            )
            output = model.text_decoder.sequence_output
            
            pooled_feature = torch.mean(output, dim=1)
            
            # Store batch results
            image_embeddings_list.append(pooled_feature.cpu())
            sensitive_attributes_list.append(s_batch.cpu())
            
    # Concatenate all data once after the loop
    image_embeddings = torch.cat(image_embeddings_list)
    
    sensitive_attributes = torch.cat(sensitive_attributes_list)

    # Save the unified file
    torch.save({
        'image_embeddings': image_embeddings,
        'sensitive_attributes': sensitive_attributes,
    }, save_path)
    print(f"Saved embeddings to {save_path}")

# Process training set
process_decoder_embeddings(train_dataset, 'embedding/fairface_BLIP_decoder_train.pt')

# Process validation set
process_decoder_embeddings(val_dataset, 'embedding/fairface_BLIP_decoder_val.pt')