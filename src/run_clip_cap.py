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
if args.mode!='sfid':
    args.encoder_prune_num = 0
    args.decoder_prune_num = 0
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # Maps GPU 2 to index 0
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import sys
sys.path.append('./')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from transformers import GPT2Tokenizer
from external.clipcap import clipcap_model
from tqdm import tqdm
from IPython.display import Image 
import nltk
from external.clipcap.clipcap_utils import decide_gender, generate
import numpy as np
import random
from tqdm import tqdm
import pandas as pd 
from PIL import Image
import pickle
import json
import pickle
import clip
from evaluation import evaluate_image_captioning

threshold = args.t
# Ensure that the necessary resources are downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

text_important_indices = None
encoder_importances = None
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

    embedding = torch.load(f'embedding/fairface_ViTB32_train.pt')
    embedding_val = torch.load(f'embedding/fairface_ViTB32_val.pt')

    X_train = embedding['image_embeddings']
    y_train = embedding['sensitive_attributes'][:,1]
    X_test = embedding_val['image_embeddings']
    y_test = embedding_val['sensitive_attributes'][:,1]


    img_model_path = 'checkpoint/CLIPCAP_img_encoder_random_forest_model.joblib'
    if os.path.exists(img_model_path):
        img_clf = load(img_model_path)
    else : 
        img_clf = RandomForestClassifier(n_estimators=100)
        img_clf.fit(X_train, y_train)
        dump(img_clf,img_model_path)
    probabilities = img_clf.predict_proba(X_test)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = X_test[max_probabilities < threshold]    
    encoder_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
    # Identify important features (top N, for example top 10%)
    encoder_importances = img_clf.feature_importances_
    embedding_dim = X_test.shape[1]
    
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
    embedding = torch.load(f'embedding/clip_cap_decoder_fairface_train.pt')
    embedding_val = torch.load(f'embedding/clip_cap_decoder_fairface_test.pt')

    X_train = embedding['hidden_states']
    y_train = embedding['sensitive_attributes']
    X_test = embedding_val['hidden_states']
    y_test = embedding_val['sensitive_attributes']
    text_model_path = f'checkpoint/CLIPCAP_text_decoder_random_forest_model.joblib'
    if os.path.exists(text_model_path):
        dec_clf = load(text_model_path)
        print("Load pretrained Random Forest.")
    else : 
        dec_clf = RandomForestClassifier(n_estimators=100)
        dec_clf.fit(X_train, y_train)
        dump(dec_clf,text_model_path)
    
    probabilities = dec_clf.predict_proba(X_test)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = X_test[max_probabilities < threshold]    
    decoder_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
    decoder_importances = dec_clf.feature_importances_
    embedding_dim = X_test.shape[1]




seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
prefix_length = 10
model_path = 'external/clipcap/clip_cap_coco_weight.pt'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = clipcap_model.ClipCaptionModel(prefix_length, device=device)
model.load_state_dict(torch.load(model_path),strict=False)

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model = model.eval() 
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

imid_2_gender = pickle.load(open('external/clipcapval_imid_gender.pkl','rb'))
# Assuming imid_2_gender is defined
filtered_image_ids = set(imid_2_gender.keys())

# Filter the id_to_captions dictionary
filtered_id_to_captions = {image_id: captions for image_id, captions in id_to_captions.items() if image_id in filtered_image_ids}
# Main loop
results = []
remove_id = pd.read_csv("external/BLIP/remove_df.csv")
remove_id = remove_id['remove_id']


text_important_indices=None
text_mean_features_lowconfidence=None
results_filename = f"external/clipcap/result/clip_cap_{args.mode}_{str(args.target)}_{args.encoder_prune_num}_{args.decoder_prune_num}_{args.t}_features.csv"

# Check if the results file already exists
if not os.path.exists(results_filename):
    for image_id, gt_captions in tqdm(filtered_id_to_captions.items()):
        
        with torch.no_grad():
            if image_id in remove_id:
                continue
            
            image_path = os.path.join(args.image_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            image = Image.open(image_path).convert('RGB')
            ground_truth_gender = imid_2_gender[image_id]
            
                
            prefix = clip_model.encode_image(preprocess(image).unsqueeze(0).to(device)).float()
            if 'encoder' in args.target: 
                pruning_num = args.encoder_prune_num
                img_important_indices = np.argsort(encoder_importances)[-pruning_num:]  
                img_important_indices = torch.tensor(img_important_indices).to(device)
                img_mean_features_lowconfidence = torch.tensor(encoder_mean_features_lowconfidence).to(device)
                prefix[:,img_important_indices] = img_mean_features_lowconfidence[img_important_indices]
            
            if 'decoder' in args.target: 
                pruning_num = args.decoder_prune_num
                text_important_indices = np.argsort(decoder_importances)[-pruning_num:]  
                text_important_indices = torch.tensor(text_important_indices).to(device)
                text_mean_features_lowconfidence = torch.tensor(decoder_mean_features_lowconfidence).to(device)
            
            generated_text = generate(model, tokenizer, text_important_indices, text_mean_features_lowconfidence, embed=prefix, mode=args.mode)
            
            # Detect gender in the generated text
            detected_gender = decide_gender(nltk.word_tokenize(generated_text))

            # Store the results
            results.append({
                'image_id': image_id,
                'ground_truth_gender': ground_truth_gender,
                'detected_gender': detected_gender,
                'gt_captions': gt_captions,
                'generated_text': generated_text
            })
    df = pd.DataFrame(results)
    df.to_csv(results_filename, index=False)
        
else:
    print(f"Results file {results_filename} already exists. Skipping processing.")


evaluate_image_captioning(results_filename)