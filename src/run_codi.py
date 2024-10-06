
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='+', default=['decoder'])
parser.add_argument('--mode', default='sfid', type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--encoder_num', default=50, type=int)
parser.add_argument('--decoder_num', default=50, type=int)
parser.add_argument('--t', default=0.9, type=float)
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # Maps GPU 2 to index 0
threshold = args.t
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import warnings
warnings.filterwarnings('ignore')
original_cwd = os.getcwd()
codi_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'codi'))
os.chdir(codi_dir)
import sys
sys.path.insert(0, codi_dir)
print("Current working directory:", os.getcwd())

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from evaluation_t2i import evaluate_t2i
encoder_debias = False
decoder_debias = False

if 'encoder' in args.target:
    encoder_debias = True
if 'decoder' in args.target:
    decoder_debias = True

text_important_indices_enc = None 
text_enc_results_dict = None
text_mean_features_lowconfidence_enc = None
image_important_indices_dec = None 
image_important_indices_dec_down = None
image_important_indices_dec_mid = None
image_mean_features_lowconfidence_dec_down = None
image_mean_features_lowconfidence_dec_mid = None
image_dec_results_dict_mid = None
image_dec_results_dict_down = None
directory_path = f"result/result_codi/{args.mode}_encoder_prune_r{args.encoder_num}_{args.t}"
if decoder_debias:
    if encoder_debias:
        directory_path = f"result/result_codi/{args.mode}_joint_prune_E{args.encoder_num}_D{args.decoder_num}_{args.t}"
    else : 
        directory_path = f"result/result_codi/{args.mode}_decoder_prune_r{args.decoder_num}_{args.t}"
run_code = True
if os.path.exists(directory_path):
    if len(os.listdir(directory_path)) > 100:
        run_code = False
        
    else:
        run_code = True
if run_code:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    from core.models.model_module_infer import model_module

    model_load_paths = ['CoDi_encoders.pth',  'CoDi_video_diffuser_8frames.pth']
    inference_tester = model_module(data_dir='../../checkpoint/', pth=model_load_paths, fp16=True)  # turn on fp16=True if loading fp16 weights

    inference_tester = inference_tester.cuda()
    inference_tester = inference_tester.eval()

    with open('profession.txt', 'r') as file:
            # Read the lines of the file into a list
        preofession = file.readlines()
    if args.mode!='sfid':
        args.target=None
        
    if 'encoder' in args.target:
        embedding = torch.load(f'../../embedding/codi_bios_train_dataset_full.pt')
        embedding_val = torch.load(f'../../embedding/codi_bios_val_dataset_full.pt')
        X_train = embedding['text_embedding']
        y_train = embedding['gender']
        X_test = embedding_val['text_embedding']
        y_test = embedding_val['gender']
        text_model_path = '../../checkpoint/codi_encoder_random_forest_model.joblib'
        
        if os.path.exists(text_model_path):
            print('RandomForest Exist!')
            text_clf = load(text_model_path)
        else : 
            print('Train RandomForest!')
            text_clf = RandomForestClassifier(n_estimators=100)
            text_clf.fit(X_train, y_train)
            dump(text_clf,text_model_path)
        probabilities = text_clf.predict_proba(X_test)
        max_probabilities = probabilities.max(axis=1)
        low_confidence_samples = X_test[max_probabilities < threshold]    
        text_mean_features_lowconfidence_enc = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
        importances = text_clf.feature_importances_
        embedding_dim = X_test.shape[1]
        
        pruning_num = int(args.encoder_num)
        text_important_indices_enc = np.argsort(importances)[-pruning_num:] 
        text_important_indices_enc = torch.tensor(text_important_indices_enc).to(device)
        text_mean_features_lowconfidence_enc = torch.tensor(text_mean_features_lowconfidence_enc).to(device).half()
        encoder_debias = True

        ## New

        # Probabilities for high confidence samples for label 1 (e.g., male)
        high_conf_label_1 = (probabilities[:, 1] > threshold)
        high_conf_samples_label_1 = X_test[high_conf_label_1]

        # Probabilities for high confidence samples for label 0 (e.g., female)
        high_conf_label_0 = (probabilities[:, 0] > threshold)
        high_conf_samples_label_0 = X_test[high_conf_label_0]

        
        # Calculate the mean for each category
        text_mean_features_label_1 = torch.mean(torch.tensor(high_conf_samples_label_1).float(), axis=0)
        text_mean_features_label_0 = torch.mean(torch.tensor(high_conf_samples_label_0).float(), axis=0)

        # Save the results in a dictionary
        text_enc_results_dict = {
            'man': text_mean_features_label_1.to(device).half(),
            'woman': text_mean_features_label_0.to(device).half(),
        }

    if 'decoder' in args.target:
        
        embedding = torch.load(f'../../embedding/codi_bios_train_dataset_full.pt')
        embedding_val = torch.load(f'../../embedding/codi_bios_val_dataset_full.pt')
        X_train_mid = embedding['mid_image_embedding']
        X_train_down = embedding['down_image_embedding']
        y_train = embedding['gender']
        
        X_test_mid = embedding_val['mid_image_embedding']
        X_test_down = embedding_val['down_image_embedding']
        y_test = embedding_val['gender']
        # print(X_train_mid.shape)
        # print(X_test_mid.shape)
        # print(X_train_down.shape)
        # print(X_test_down.shape)


        text_model_path_mid = '../../checkpoint/codi_decoder_random_forest_model_mid.joblib'
        text_model_path_down = '../../checkpoint/codi_decoder_random_forest_model_down.joblib'
        if os.path.exists(text_model_path_down):
            print('RandomForest Exist!')
            text_clf_down = load(text_model_path_down)
        else : 
            print('Train RandomForest!')
            text_clf_down = RandomForestClassifier(n_estimators=100)
            text_clf_down.fit(X_train_down, y_train)
            dump(text_clf_down,text_model_path_down)

        if os.path.exists(text_model_path_mid):
            print('RandomForest Exist!')
            text_clf_mid = load(text_model_path_mid)
        else : 
            print('Train RandomForest!')
            text_clf_mid = RandomForestClassifier(n_estimators=100)
            text_clf_mid.fit(X_train_mid, y_train)
            dump(text_clf_mid,text_model_path_mid)
        probabilities_mid = text_clf_mid.predict_proba(X_test_mid)
        max_probabilities_mid = probabilities_mid.max(axis=1)
        low_confidence_samples_mid = X_test_mid[max_probabilities_mid < threshold]    
        image_mean_features_lowconfidence_dec_mid = torch.mean(torch.tensor(low_confidence_samples_mid).float(),axis=0)
        importances_mid = text_clf_mid.feature_importances_
        embedding_dim = X_test_mid.shape[1]
 
        pruning_num = int(args.decoder_num)
        image_important_indices_dec_mid = np.argsort(importances_mid)[-pruning_num:] 
        image_important_indices_dec_mid = torch.tensor(image_important_indices_dec_mid).to(device)
        image_mean_features_lowconfidence_dec_mid = torch.tensor(image_mean_features_lowconfidence_dec_mid).to(device).half()


        probabilities_down = text_clf_down.predict_proba(X_test_down)
        max_probabilities_down = probabilities_down.max(axis=1)
        low_confidence_samples_down = X_test_down[max_probabilities_down < threshold]    
        image_mean_features_lowconfidence_dec_down = torch.mean(torch.tensor(low_confidence_samples_down).float(),axis=0)
        importances_down = text_clf_down.feature_importances_
        embedding_dim = X_test_down.shape[1]

        pruning_num = int(args.decoder_num)
        image_important_indices_dec_down = np.argsort(importances_down)[-pruning_num:] 
        image_important_indices_dec_down = torch.tensor(image_important_indices_dec_down).to(device)
        image_mean_features_lowconfidence_dec_down = torch.tensor(image_mean_features_lowconfidence_dec_down).to(device).half()
        decoder_debias = True

        ## New

        # Probabilities for high confidence samples for label 1 (e.g., male)
        high_conf_label_1_down = (probabilities_down[:, 1] > threshold)
        high_conf_samples_label_1_down = X_test_down[high_conf_label_1_down]

        # Probabilities for high confidence samples for label 0 (e.g., female)
        high_conf_label_0_down = (probabilities_down[:, 0] > threshold)
        high_conf_samples_label_0_down = X_test_down[high_conf_label_0_down]

        
        # Calculate the mean for each category
        image_mean_features_label_1_down = torch.mean(torch.tensor(high_conf_samples_label_1_down).float(), axis=0)
        image_mean_features_label_0_down= torch.mean(torch.tensor(high_conf_samples_label_0_down).float(), axis=0)

        # Save the results in a dictionary
        image_dec_results_dict_down = {
            'man': image_mean_features_label_1_down.to(device).half(),
            'woman': image_mean_features_label_0_down.to(device).half(),
        }
        
        ## New
        # Probabilities for high confidence samples for label 1 (e.g., male)
        high_conf_label_1_mid = (probabilities_mid[:, 1] > threshold)
        high_conf_samples_label_1_mid = X_test_mid[high_conf_label_1_mid]

        # Probabilities for high confidence samples for label 0 (e.g., female)
        high_conf_label_0_mid = (probabilities_mid[:, 0] > threshold)
        high_conf_samples_label_0_mid = X_test_mid[high_conf_label_0_mid]

        
        # Calculate the mean for each category
        image_mean_features_label_1_mid = torch.mean(torch.tensor(high_conf_samples_label_1_mid).float(), axis=0)
        image_mean_features_label_0_mid= torch.mean(torch.tensor(high_conf_samples_label_0_mid).float(), axis=0)
        # print(image_mean_features_label_1_mid.shape)
        # print(image_mean_features_label_0_mid.shape)
        # Save the results in a dictionary

        image_dec_results_dict_mid = {
            'man': image_mean_features_label_1_mid.to(device).half(),
            'woman': image_mean_features_label_0_mid.to(device).half(),
        }


    # Remove any trailing newline characters from each line
    preofession = [line.rstrip('\n') for line in preofession]
    all_list = []
    genders = ['a person', 'a man','a woman']
    cnt = 0
    results = []  
    
    for seed in range(10):
        for gender in genders:
            for job in preofession:
                if job[0] in ['a','e','i','o','u']:
                    article = 'an'
                else :
                    article = 'a'
                # prompt= f'A real human photo of {gender} who works as {article} {job}'
                prompt= f'{gender} who works as {article} {job}'
                images = inference_tester.inference(
                                xtype = ['image'],
                                condition = [prompt],
                                condition_types = ['text'],
                                n_samples = 1, 
                                image_size = 256,
                                ddim_steps = 50,
                                encoder_debias=encoder_debias, 
                                decoder_debias = decoder_debias,
                                gender=gender,
                                # text_important_indices_enc=text_important_indices_enc,
                                # text_mean_features_lowconfidence_enc=text_mean_features_lowconfidence_enc,
                                # image_important_indices_dec=(image_important_indices_dec_down,image_important_indices_dec_mid) ,
                                # image_mean_features_lowconfidence_dec=(image_mean_features_lowconfidence_dec_down,image_mean_features_lowconfidence_dec_mid),
                                text_important_indices_enc=text_important_indices_enc,
                                text_enc_results_dict=text_enc_results_dict,
                                image_important_indices_dec=(image_important_indices_dec_down,image_important_indices_dec_mid) ,
                                image_dec_results_dict=(image_dec_results_dict_down,image_dec_results_dict_mid),
                                )
                
                result = images[0][0]    
                image_path = os.path.join(directory_path, f"{job}_{gender}_{seed}.png")
                result.save(image_path)


            

evaluate_t2i(directory_path,device=device)