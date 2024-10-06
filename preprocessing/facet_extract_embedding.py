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

new_annotations = pd.DataFrame(columns=['filename', 'class1', 'gender', 'race', 'age'])
annotations = pd.read_csv('../data/facet/annotations/annotations.csv')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_list = ['RN50', 'ViT-B/32']
print(f"Extract CLIP image and text embedding for FACET zero-shot classification with baseline models {model_list}.")
print("Load annotations...")
# Iterate through the annotations and process each image
for index, row in tqdm(annotations.iterrows(), total=len(annotations)):
    
    class1 = row['class1']
    # Check gender presentation and assign the gender attribute accordingly
    if row['gender_presentation_non_binary'] == 1 or row['gender_presentation_na'] == 1:
        continue  # Skip non-binary or NA genders
    gender = 0 if row['gender_presentation_masc'] == 1 else 1
    # Check race presentation and assign the race attribute accordingly
    if row['skin_tone_1'] > 0 or row['skin_tone_2'] > 0:
        race = 0
    elif row['skin_tone_4'] > 0 or row['skin_tone_5'] > 0 or row['skin_tone_6'] > 0:
        race = 1
    elif row['skin_tone_9'] > 0 or row['skin_tone_10'] > 0:
        race = 2
    else:
        continue
    # Check age presentation and assign the age attribute accordingly
    if row['age_presentation_young'] == 1:
        age = 0
    elif row['age_presentation_middle'] == 1:
        age = 0
    elif row['age_presentation_older'] == 1:
        age = 1
    else:
        continue  # Skip other age presentations
    new_row = {'filename': row['filename'], 'class1': class1, 'gender': gender, 'race': race, 'age': age}
    new_annotations = pd.concat([new_annotations, pd.DataFrame([new_row])], ignore_index=True)

for clip_model in model_list:
    print(f"Model {clip_model}.")
    base_model, preprocess = clip.load(clip_model, device=device)
    base_model.eval()
    output_dim = 512  # Adjust output_dim based on your model
    image_dir = '../data/facet/image/'
    prefix = clip_model.replace("/", "").replace('-', '')

    # Define paths for the files
    train_save_path = f'embedding/{prefix}_facet_train_data_dict.pt'
    val_save_path = f'embedding/{prefix}_facet_val_data_dict.pt'

    # Check if the files already exist
    if os.path.exists(train_save_path) and os.path.exists(val_save_path):
        print(f"Files already exist at: {train_save_path} and {val_save_path}. Skipping.")
        continue

    # Initialize empty lists to store embeddings and attributes
    embeddings = []
    genders = []
    classes = []
    races = []
    ages = []
    filenames = []

    # Iterate through the annotations and process each image and caption
    for index, row in tqdm(new_annotations.iterrows(), total=len(new_annotations)):
        image_path = os.path.join(image_dir, row['filename'])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)

        # Get the image embedding from the CLIP model
        with torch.no_grad():
            image_embedding = base_model.encode_image(image)

        # Normalize the image embedding
        image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Store the normalized embeddings in the lists
        embeddings.append(image_embedding_norm.squeeze(0).cpu())
        genders.append(row['gender'])
        classes.append(row['class1'])
        races.append(row['race'])
        ages.append(row['age'])
        filenames.append(row['filename'])

    # Split the data into training and validation sets (80% train, 20% validation)
    train_indices, val_indices = train_test_split(range(len(embeddings)), test_size=0.2, random_state=0)

    # Create dictionaries to store the training and validation data
    train_data_dict = {
        'image_embedding': torch.stack([embeddings[i] for i in train_indices]),
        'gender': torch.tensor([genders[i] for i in train_indices]),
        'class1': [classes[i] for i in train_indices],
        'race': torch.tensor([races[i] for i in train_indices]),
        'age': torch.tensor([ages[i] for i in train_indices]),
        'filename': [filenames[i] for i in train_indices]
    }

    val_data_dict = {
        'image_embedding': torch.stack([embeddings[i] for i in val_indices]),
        'gender': torch.tensor([genders[i] for i in val_indices]),
        'class1': [classes[i] for i in val_indices],
        'race': torch.tensor([races[i] for i in val_indices]),
        'age': torch.tensor([ages[i] for i in val_indices]),
        'filename': [filenames[i] for i in val_indices]
    }

    # Save the training and validation data dictionaries to files
    torch.save(train_data_dict, train_save_path)
    torch.save(val_data_dict, val_save_path)
    print(f'Files saved at: {train_save_path} and {val_save_path}')
