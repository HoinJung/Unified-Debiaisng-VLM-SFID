
import os
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random   
from joblib import dump, load
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--target', nargs='+', default=['image'])
parser.add_argument('--base', default='ViT-B/32', type=str)
parser.add_argument('--mode', default='sfid', type=str)
parser.add_argument('--image_prune_num', default=100, type=int)
parser.add_argument('--text_prune_num', default=100, type=int)
parser.add_argument('--t', default=0.7, type=float)

args = parser.parse_args()

clip_model = args.base
clip_name = clip_model.replace("/", "").replace('-', '')
threshold = args.t
device = "cuda" if torch.cuda.is_available() else "cpu"
from evaluation import evaluate_facet, evaluate_flickr


import time
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


img_important_indices = None
img_mean_features_lowconfidence = None

if 'image' in args.target:
    print("Debias Image Encoder")    
    embedding = torch.load(f'embedding/fairface_{clip_name}_train.pt')
    embedding_val = torch.load(f'embedding/fairface_{clip_name}_val.pt')
    X_train = embedding['image_embeddings']
    y_train = embedding['sensitive_attributes'][:,1]
    X_test = embedding_val['image_embeddings']
    y_test = embedding_val['sensitive_attributes'][:,1]
    img_model_path = f'checkpoint/{clip_name}_img_encoder_random_forest_model.joblib'
    if os.path.exists(img_model_path):
        img_clf = load(img_model_path)
        print("Load pretrained Random Forest.")
    else : 
        img_clf = RandomForestClassifier(n_estimators=100)
        img_clf.fit(X_train, y_train)
        dump(img_clf,img_model_path)
    probabilities = img_clf.predict_proba(X_test)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = X_test[max_probabilities <threshold]
    
    importances = img_clf.feature_importances_
    img_important_indices =  torch.tensor(np.argsort(importances)[-args.image_prune_num:] ).to(device)
    img_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0).to(device)
    

text_important_indices = None
text_mean_features_lowconfidence = None 

if 'text' in args.target:
    print("Debias Text Encoder")
    embedding = torch.load(f'embedding/{clip_name}_bios_bias_text_train.pt')
    embedding_val = torch.load(f'embedding/{clip_name}_bios_bias_text_val.pt')
    X_train = embedding['text_embeddings'].cpu()
    y_train = embedding['sensitive_attributes'].cpu()
    X_test = embedding_val['text_embeddings'].cpu()
    y_test = embedding_val['sensitive_attributes'].cpu()
    text_model_path = f'checkpoint/{clip_name}_text_encoder_random_forest_model.joblib'
    if os.path.exists(text_model_path):
        text_clf = load(text_model_path)
        print("Load pretrained Random Forest.")
    else : 
        text_clf = RandomForestClassifier(n_estimators=100)
        text_clf.fit(X_train, y_train)
        dump(text_clf,text_model_path)
    probabilities = text_clf.predict_proba(X_test)
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = X_test[max_probabilities < threshold]    
    
    importances = text_clf.feature_importances_
    text_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0).to(device)
    text_important_indices = torch.tensor(np.argsort(importances)[-args.text_prune_num:] ).to(device)


print("*"*10, f"Evaluate FACET","*"*10 )
evaluate_facet(args,clip_model,device,clip_name,img_important_indices=img_important_indices,img_mean_features_lowconfidence=img_mean_features_lowconfidence\
                ,text_important_indices=text_important_indices,text_mean_features_lowconfidence=text_mean_features_lowconfidence)


print("*"*10, f"Evaluate Flickr","*"*10 )
evaluate_flickr(args,clip_model,device,clip_name,img_important_indices=img_important_indices,img_mean_features_lowconfidence=img_mean_features_lowconfidence\
                ,text_important_indices=text_important_indices,text_mean_features_lowconfidence=text_mean_features_lowconfidence)
