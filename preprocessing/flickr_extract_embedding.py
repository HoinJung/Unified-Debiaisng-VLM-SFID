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
import re

model_list = ['RN50', 'ViT-B/32']
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Extract CLIP image and text embedding for Flickr30K text-to-image retrieval with baseline models {model_list}.")
def preprocess_caption(caption):
    caption = caption.lower()
    gender_neutral_map = {
        r'\bman\b': 'person',
        r'\bwoman\b': 'person',
        r'\bmen\b': 'people',
        r'\bwomen\b': 'people',
        r'\bmale\b': '',
        r'\bfemale\b': '',
        r'\bmales\b': 'people',
        r'\bfemales\b': 'people',
        r'\bguys\b': 'people',
        r'\bladies\b': 'people',
        r'\bguy\b': 'person',
        r'\blady\b': 'person',
        r"\bmen and women\b": 'people',
        r'\bgirl\b': 'child',
        r'\bson\b': 'child',
        r'\bboy\b': 'child',
        r'\bdad\b': 'person',
        r'\bgirls\b': 'children',
        r'\bboys\b': 'children',
        r'\bmom\b': 'person',
        r'\bmother\b': 'person',
        r'\bsister\b': 'person',
        r'\bdaughter\b': 'child',
        r'\bwife\b': 'person',
        r'\bgirlfriend\b': 'person',
        r'\bgentleman\b': 'person',
        r'\bfather\b': 'person',
        r'\bbrother\b': 'person',
        r'\bhusband\b': 'person',
        r'\bboyfriend\b': 'person',
        ' , ': ', '
    }
    for pattern, replacement in gender_neutral_map.items():
        caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
    caption = caption.strip('.').strip() + '.'
    return caption

def determine_gender(caption):
    male_words = [
        r'\bman\b', r'\bmen\b', r'\bmale\b', r'\bguy\b', r'\bson\b', r'\bboy\b',
        r'\bgentleman\b', r'\bfather\b', r'\bbrother\b', r'\bhusband\b', r'\bboyfriend\b',
        r'\bmales\b', r'\bboys\b', r'\bguys\b', r'\bbrothers\b'
    ]
    female_words = [
        r'\bwoman\b', r'\bwomen\b', r'\bfemale\b', r'\blady\b', r'\bladies\b',
        r'\bgirl\b', r'\bmom\b', r'\bmother\b', r'\bsister\b', r'\bdaughter\b',
        r'\bwife\b', r'\bgirlfriend\b', r'\bfemales\b', r'\bgirls\b', r'\bsisters\b',
        r'\bdaughters\b'
    ]
    caption_lower = caption.lower()
    if any(re.search(word, caption_lower) for word in male_words):
        return 0
    elif any(re.search(word, caption_lower) for word in female_words):
        return 1
    else:
        return -1

def create_dataset(version):
    # Define the file paths
    image_file_path = f'embedding/flickr/flickr_1000images_v{version}.csv'
    caption_file_path = f'embedding/flickr/flickr_with_gender_neutral_captions_v{version}.csv'

    # Check if the files already exist
    if os.path.exists(image_file_path) and os.path.exists(caption_file_path):
        print(f"Files already exist for version {version} at: {image_file_path} and {caption_file_path}. Skipping.")
        return

    # flickr30k_images
    df = pd.read_csv("../data/flickr/results.csv", delimiter="|")
    df.columns = ['image', 'caption_number', 'caption']
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
    ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
    df['id'] = ids

    df['neutral_caption'] = df['caption'].apply(preprocess_caption)
    df['gender'] = df['caption'].apply(determine_gender)

    def most_frequent_gender(x):
        return x.value_counts().idxmax()

    df_final = df.groupby(['image', 'id'])['gender'].apply(most_frequent_gender).reset_index()

    # Filter the rows where gender is 0
    gender_0 = df_final[df_final['gender'] == 0].sample(n=500, random_state=version)

    # Filter the rows where gender is 1
    gender_1 = df_final[df_final['gender'] == 1].sample(n=500, random_state=version)

    # Concatenate both dataframes
    sampled_df = pd.concat([gender_0, gender_1])

    sampled_df.reset_index(drop=True, inplace=True)

    # Save the dataframes to CSV files
    sampled_df.to_csv(image_file_path, index=False)
    df.to_csv(caption_file_path, index=False)

    print(f"Files saved for version {version} at: {image_file_path} and {caption_file_path}")

print("Choose 1000 random images with balanced gender.")
print("Create 10 different version for confidence interval.")
for i in range(10):
    create_dataset(i)

def process_version(version, clip_model, model, preprocess):
    # Paths for saving
    text_save_path = f'embedding/flickr1000_{clip_model.replace("/", "-").replace("-", "")}_text_v{version}.pt'
    image_save_path = f'embedding/flickr1000_{clip_model.replace("/", "-").replace("-", "")}_image_v{version}.pt'

    # Check if the files already exist
    if os.path.exists(text_save_path) and os.path.exists(image_save_path):
        print(f"Files already exist for version {version} at: {text_save_path} and {image_save_path}. Skipping.")
        return

    # Load the dataframes
    caption_df = pd.read_csv(f'embedding/flickr/flickr_with_gender_neutral_captions_v{version}.csv')
    image_df = pd.read_csv(f'embedding/flickr/flickr_1000images_v{version}.csv')

    caption_df = caption_df[caption_df['id'].isin(image_df['id'])]
    caption_df = pd.merge(caption_df, image_df[['id', 'gender']], on='id', how='left')

    # Truncate captions to fit within CLIP's token limit
    max_tokens = 77
    captions = caption_df['neutral_caption'].tolist()
    genders = caption_df['gender_y'].tolist()
    ids = caption_df['id'].tolist()
    truncated_captions = []
    for caption in captions:
        tokens = clip.tokenize([caption], truncate=True).squeeze(0)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        truncated_captions.append(tokens)

    batch_size = 512

    text_features = []
    all_genders = []
    all_ids = []
    
    for i in range(0, len(truncated_captions), batch_size):
        batch_captions = truncated_captions[i:i + batch_size]
        batch_genders = genders[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        text_inputs = torch.stack(batch_captions).to(device)
        with torch.no_grad():
            batch_text_features = model.encode_text(text_inputs)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)
        text_features.append(batch_text_features.cpu())
        all_genders.extend(batch_genders)
        all_ids.extend(batch_ids)

    text_features = torch.cat(text_features)
    torch.save({"text_embeddings": text_features, 'gender': all_genders, 'id': all_ids}, text_save_path)

    all_genders = []
    all_ids = []
    image_names = image_df['image'].tolist()
    genders = image_df['gender'].tolist()
    ids = image_df['id'].tolist()
    image_features = []
    for i in range(0, len(image_names), batch_size):
        batch_images = []
        batch_genders = genders[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        for image_name in image_names[i:i + batch_size]:
            image_path = f'../data/flickr/flickr30k_images/{image_name}'
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            batch_images.append(image)
        batch_images = torch.cat(batch_images)
        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            image_features.append(batch_image_features.cpu())
        all_genders.extend(batch_genders)
        all_ids.extend(batch_ids)

    image_features = torch.cat(image_features)
    torch.save({'image_embeddings': image_features, 'gender': all_genders, 'id': all_ids}, image_save_path)

for clip_model in model_list:
    print(f'Extract embedding for model {clip_model}')
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    for i in tqdm(range(10)):
        process_version(i, clip_model, model, preprocess)
