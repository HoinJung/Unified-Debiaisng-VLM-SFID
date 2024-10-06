#@title Imports
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='+', default=['encoder'])
parser.add_argument('--mode', default='sfid', type=str)
parser.add_argument('--gpu_id', default='2', type=str)
parser.add_argument('--encoder_prune_num', default=50, type=int)
parser.add_argument('--decoder_prune_num', default=50, type=int)
parser.add_argument('--t', default=0.9, type=float)
parser.add_argument('--pred_cap_path', default="external/clipcap/oscar_preds.pkl", type=str)
parser.add_argument('--image_dir', default="../data/COCO/val2014", type=str)

args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # Maps GPU 2 to index 0
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import sys
sys.path.append('./')
sys.path.append('external/BLIP/')
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import nltk
import numpy as np
import random
import json
import pickle
from external.clipcap.clipcap_utils import decide_gender
from external.BLIP.models.blip import blip_decoder
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from evaluation import evaluate_image_captioning

# Ensure that the necessary resources are downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

threshold = args.t
model_path = 'external/BLIP/model_base_capfilt_large.pth'
image_size = 384
model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

with open('../data/COCO/annotations/captions_val2014.json', 'r') as json_data:
    d = json.load(json_data)
annotations = d['annotations']

# Create a dictionary mapping image IDs to captions
id_to_captions = {}
for ann in annotations:
    image_id = ann['image_id']
    caption = ann['caption']
    if image_id not in id_to_captions:
        id_to_captions[image_id] = []
    id_to_captions[image_id].append(caption)
imid_2_gender = pickle.load(open('external/clipcap/Data/val_imid_gender.pkl','rb'))
# Assuming imid_2_gender is defined
filtered_image_ids = set(imid_2_gender.keys())

# Filter the id_to_captions dictionary
filtered_id_to_captions = {image_id: captions for image_id, captions in id_to_captions.items() if image_id in filtered_image_ids}

encoder_mean_features_lowconfidence=None
encoder_importances=None
decoder_mean_features_lowconfidence=None
decoder_importances=None
encoder_debias=False
decoder_debias=False

    
results_filename = f"external/BLIP/result/blip_{args.mode}_{str(args.target)}_{args.encoder_prune_num}_{args.decoder_prune_num}_{args.t}_features.csv"
remove_id = pd.read_csv("external/BLIP/remove_df.csv")
remove_id = remove_id['remove_id']

if not os.path.exists(results_filename):
    if 'encoder' in args.target:
        print("Debias Image Encoder")
        sfid_seed = 0
        random.seed(sfid_seed)
        torch.manual_seed(sfid_seed)
        torch.cuda.manual_seed(sfid_seed)
        np.random.seed(sfid_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmarks=False
        os.environ['PYTHONHASHSEED'] = str(sfid_seed)

        embedding = torch.load(f'embedding/fairface_BLIP_encoder_train.pt')
        embedding_val = torch.load(f'embedding/fairface_BLIP_encoder_val.pt')

        X_train = embedding['image_embeddings']
        y_train = embedding['sensitive_attributes'][:,1]
        X_test = embedding_val['image_embeddings']
        y_test = embedding_val['sensitive_attributes'][:,1]


        enc_model_path = 'checkpoint/BLIP_encoder_random_forest_model.joblib'
        if os.path.exists(enc_model_path):
            enc_clf = load(enc_model_path)
        else : 
            enc_clf = RandomForestClassifier(n_estimators=100)
            enc_clf.fit(X_train, y_train)
            dump(enc_clf,enc_model_path)

        probabilities = enc_clf.predict_proba(X_test)
        max_probabilities = probabilities.max(axis=1)
        low_confidence_samples = X_test[max_probabilities < threshold]
        encoder_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
        
        # Identify important features (top N, for example top 10%)
        encoder_importances = enc_clf.feature_importances_
        embedding_dim = X_test.shape[1]
        encoder_debias=True
    if 'decoder' in args.target:
        print("Debias Text Decoder")
        sfid_seed = 0
        random.seed(sfid_seed)
        torch.manual_seed(sfid_seed)
        torch.cuda.manual_seed(sfid_seed)
        np.random.seed(sfid_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmarks=False
        os.environ['PYTHONHASHSEED'] = str(sfid_seed)

        embedding = torch.load(f'embedding/fairface_BLIP_decoder_train.pt')
        embedding_val = torch.load(f'embedding/fairface_BLIP_decoder_val.pt')

        X_train = embedding['image_embeddings']
        y_train = embedding['sensitive_attributes'][:,1]
        X_test = embedding_val['image_embeddings']
        y_test = embedding_val['sensitive_attributes'][:,1]


        dec_model_path = 'checkpoint/BLIP_decoder_random_forest_model.joblib'
        if os.path.exists(dec_model_path):
            dec_clf = load(dec_model_path)
        else : 
            dec_clf = RandomForestClassifier(n_estimators=100)
            dec_clf.fit(X_train, y_train)
            dump(dec_clf,dec_model_path)
        

        probabilities = dec_clf.predict_proba(X_test)
        max_probabilities = probabilities.max(axis=1)
        low_confidence_samples = X_test[max_probabilities < threshold]   
        decoder_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
        
        # Identify important features (top N, for example top 10%)
        decoder_importances = dec_clf.feature_importances_
        embedding_dim = X_test.shape[1]
        decoder_debias=True


    results = []

    transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Check if the results file already exists



    for image_id, gt_captions in tqdm(filtered_id_to_captions.items()):
        if image_id in remove_id:
            continue
        
        image_path = os.path.join(args.image_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
        raw_image = Image.open(image_path).convert('RGB')
        ground_truth_gender = imid_2_gender[image_id]

        # Generate caption
        image = transform(raw_image).unsqueeze(0).to(device)

        with torch.no_grad():
        
            generated_text = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5,\
                                            encoder_debias=encoder_debias, decoder_debias=decoder_debias,debias_arg=args,device=device,\
                                                encoder_mean_features_lowconfidence=encoder_mean_features_lowconfidence,\
                                                    encoder_importances=encoder_importances,\
                                                        decoder_mean_features_lowconfidence=decoder_mean_features_lowconfidence,\
                                                    decoder_importances=decoder_importances)[0]
        
        # Detect gender in the generated text
        detected_gender = decide_gender(nltk.word_tokenize(generated_text))

        results.append({
            'image_id': image_id,
            'ground_truth_gender': ground_truth_gender,
            'detected_gender': detected_gender,
            'gt_captions':gt_captions,
            'generated_text':generated_text,
        })
    df = pd.DataFrame(results)
    df.to_csv(results_filename, index=False)
else:
    print(f"Results file {results_filename} already exists. Skipping processing.")


evaluate_image_captioning(results_filename)