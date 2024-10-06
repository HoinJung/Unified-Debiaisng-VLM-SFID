import torch
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import Dataset
import torch

class FacetDataset(Dataset):
    def __init__(self, clip_name, device, class_to_idx):
        # Load train data
        train_file = torch.load(f'embedding/{clip_name}_facet_train_data_dict.pt')
        val_file = torch.load(f'embedding/{clip_name}_facet_val_data_dict.pt')

        # Combine train and val data
        # Assuming data is in PyTorch tensors
        self.image_embedding = torch.cat((train_file['image_embedding'], val_file['image_embedding']), dim=0)
        self.gender = torch.cat((train_file['gender'], val_file['gender']), dim=0)
        self.class_names = train_file['class1'] + val_file['class1']
        self.race = torch.cat((train_file['race'], val_file['race']), dim=0)
        self.age = torch.cat((train_file['age'], val_file['age']), dim=0)
        self.device = device
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.image_embedding)

    def __getitem__(self, idx):
        image_embedding = torch.tensor(self.image_embedding[idx]).to(self.device).float()
        gender = torch.tensor(self.gender[idx]).to(self.device).long()
        class_name = self.class_names[idx]
        class_label = torch.tensor(self.class_to_idx[class_name]).to(self.device).long()
        race = torch.tensor(self.race[idx]).to(self.device).float()
        age = torch.tensor(self.age[idx]).to(self.device).float()
        return image_embedding, gender, race, age, class_label

    
# try :
#     embedding = torch.load(f'../../survey/data/CODI_text_decoder_manual_train.pt')
#     embedding_val = torch.load(f'../../survey/data/CODI_text_decoder_manual_val.pt')
# except:
#     embedding = torch.load(f'../../../survey/data/CODI_text_decoder_manual_train.pt')
#     embedding_val = torch.load(f'../../../survey/data/CODI_text_decoder_manual_val.pt')


def cosine_similarity(x1, x2):
    x1 = x1 / x1.norm(dim=1, keepdim=True)
    x2 = x2 / x2.norm(dim=1, keepdim=True)
    return x1 @ x2.T

def calculate_accuracy(preds, trues):
    return accuracy_score(trues, preds)


def zero_shot_classifier(image_embeddings, text_embeddings, class_labels):
    similarities = cosine_similarity(image_embeddings.float(), text_embeddings.float())
    similarities = similarities.softmax(-1)
    predictions = similarities.argmax(dim=-1)
    predicted_labels = [class_labels[pred] for pred in predictions.cpu().numpy()]
    return predicted_labels

def evaluate_gender_difference(image_embeddings, image_ids, image_genders, text_embeddings, text_ids, text_genders, top_k):
    # Calculate the similarities between images and texts
    similarities = (100 * image_embeddings @ text_embeddings.T).softmax(dim=-1)

    # Initialize counters for the number of retrieved images for each gender
    # retrieved_counts = {'male': 0, 'female': 0}
    min_skew_list = []
    max_skew_list = []
    bias_diff=[]
    # Iterate over each text and its corresponding image
    for text_index, text_id in enumerate(text_ids):
        # Find the indices of the top K most similar images
        top_k_indices = torch.topk(similarities[:, text_index], k=top_k).indices
        retrieved_image_ids = np.array(image_ids)[top_k_indices[:top_k]]
        retrieved_counts = {'male': 0, 'female': 0}
        # Count the number of retrieved images for each gender
        for image_id in retrieved_image_ids:
            image_gender = image_genders[image_ids.index(image_id)]
            gender_key = 'male' if image_gender == 0 else 'female'
            retrieved_counts[gender_key] += 1
        male_skew = np.log((retrieved_counts['male']/top_k) /0.5)  # The same number of images for each gender
        female_skew = np.log((retrieved_counts['female']/top_k) /0.5) 
        bias_diff = retrieved_counts['male'] - retrieved_counts['female']
        # min_skew = min(male_skew, female_skew)
        
        max_skew = max(abs(male_skew), abs(female_skew))
        
    # min_skew_list.append(min_skew)    
    max_skew_list.append(max_skew)    
    
    return  np.round(np.mean(max_skew_list),4)

def evaluate_recall(image_embeddings, image_ids, text_embeddings, text_ids, top_k):
    # Calculate the similarities between images and texts
    similarities = (100 * image_embeddings @ text_embeddings.T).softmax(dim=-1)

    # Initialize counters for Recall@K
    recall_counts = [0] * len(text_ids)
    
    # Iterate over each text and its corresponding image
    cont_list = []
    for text_index, text_id in enumerate(text_ids):
        # Find the indices of the top K most similar images
        top_k_indices = torch.topk(similarities[:, text_index], k=top_k).indices
        if np.isin(text_id,np.array(image_ids)[top_k_indices[:top_k]]):
            recall_counts[text_index] +=1
            
    
    # Calculate the Recall@K percentages
    recall_percentages = sum(recall_counts) /len(text_ids)

    return recall_percentages
