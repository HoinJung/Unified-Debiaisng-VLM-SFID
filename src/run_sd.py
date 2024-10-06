
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='+', default=['decoder'])
parser.add_argument('--mode', default='sfid', type=str)
parser.add_argument('--gpu_id', default='2', type=str)
parser.add_argument('--encoder_num', default=50, type=int)
parser.add_argument('--decoder_num', default=50, type=int)
parser.add_argument('--t', default=0.5, type=float)
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # Maps GPU 2 to index 0
threshold = args.t
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from evaluation_t2i import evaluate_t2i
encoder_debias = False
decoder_debias = False

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


if 'encoder' in args.target:
    encoder_debias = True
if 'decoder' in args.target:
    decoder_debias = True


directory_path = f"external/SD/result/{args.mode}_encoder_prune_r{args.encoder_num}_{args.t}"
if decoder_debias:
    if encoder_debias:
        directory_path = f"external/SD/result/{args.mode}_joint_prune_E{args.encoder_num}_D{args.decoder_num}_{args.t}"
    else : 
        directory_path = f"external/SD/result/{args.mode}_decoder_prune_r{args.decoder_num}_{args.t}"
# Check if the directory exists, if so, skip further processing

run_code = True

if os.path.exists(directory_path):
    if len(os.listdir(directory_path)) > 100:
        run_code = False
        
    else:
        run_code = True
if run_code:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    import sys
    sys.path.append("./")
    from diffusers import StableDiffusionPipeline
    from external.SD.sdxl_code import CustomStableDiffusionPipeline,CustomUNet2DConditionModel
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = CustomStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    custom_unet = CustomUNet2DConditionModel(
        # Ensure you pass all necessary arguments that the original UNet2DConditionModel requires
        in_channels=pipe1.unet.config.in_channels,
        out_channels=pipe1.unet.config.out_channels,
        layers_per_block=pipe1.unet.config.layers_per_block,
        block_out_channels=pipe1.unet.config.block_out_channels,
        down_block_types=pipe1.unet.config.down_block_types,
        up_block_types=pipe1.unet.config.up_block_types,
        attention_head_dim=pipe1.unet.config.attention_head_dim,
        cross_attention_dim=pipe1.unet.config.cross_attention_dim,
        use_linear_projection=pipe1.unet.config.use_linear_projection,
        sample_size=pipe1.unet.config.sample_size 
    ).to(torch.float16).to(device) 

    custom_unet.load_state_dict(pipe1.unet.state_dict())
    pipe.unet = custom_unet.to(torch.float16).to(device)
    pipe = pipe.to(torch.float16).to(device)



    if 'encoder' in args.target:

        embedding = torch.load(f'embedding/sd_bios_encoder_train_dataset.pt')
        embedding_val = torch.load(f'embedding/sd_bios_encoder_val_dataset.pt')
        X_train = embedding['encoder_embedding'].detach().cpu().numpy()
        y_train = embedding['gender'].detach().cpu().numpy()
        X_test = embedding_val['encoder_embedding'].detach().cpu().numpy()
        y_test = embedding_val['gender']
        text_model_path = 'checkpoint/SD_encoder_random_forest_model.joblib'
        
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

        embedding = torch.load(f'embedding/sd_bios_decoder_train_dataset.pt')
        embedding_val = torch.load(f'embedding/sd_bios_decoder_val_dataset.pt')
        X_train_mid = embedding['decoder_mid_embedding'].detach().cpu().numpy()
        X_train_down = embedding['decoder_down_embedding'].detach().cpu().numpy()
        X_test_mid = embedding_val['decoder_mid_embedding'].detach().cpu().numpy()
        X_test_down = embedding_val['decoder_down_embedding'].detach().cpu().numpy()
        y_train = embedding['gender'].detach().cpu().numpy()
        y_test = embedding_val['gender']
        
        
        

        text_model_path_mid = 'checkpoint/SD_decoder_mid_random_forest_model.joblib'
        text_model_path_down = 'checkpoint/SD_decoder_down_random_forest_model.joblib'
        
        
        if os.path.exists(text_model_path_mid):
            print('RandomForest Exist!')
            
            text_clf_mid = load(text_model_path_mid)
        else : 
            print('Train RandomForest!')
            # X_train_mid = torch.cat(X_train_mid, dim=0)
            text_clf_mid = RandomForestClassifier(n_estimators=100)
            
            text_clf_mid.fit(X_train_mid, y_train)
            dump(text_clf_mid,text_model_path_mid)
        
        if os.path.exists(text_model_path_down):
            print('RandomForest Exist!')
            
            text_clf_down = load(text_model_path_down)
        else : 
            print('Train RandomForest!')
            # X_train_down = torch.cat(X_train_down, dim=0)
            text_clf_down = RandomForestClassifier(n_estimators=100)
            text_clf_down.fit(X_train_down, y_train)
            dump(text_clf_down,text_model_path_down)

        probabilities_mid = text_clf_mid.predict_proba(X_test_mid)
        max_probabilities_mid = probabilities_mid.max(axis=1)
        low_confidence_samples_mid = X_test_mid[max_probabilities_mid < threshold]    
        image_mean_features_lowconfidence_dec_mid = torch.mean(torch.tensor(low_confidence_samples_mid).float(),axis=0)
        importances_mid = text_clf_mid.feature_importances_
        
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
        
    with open('external/codi/profession.txt', 'r') as file:
            # Read the lines of the file into a list
        preofession = file.readlines()

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
                # prompt= f'{gender} who works as {article} {job}'
                prompt= f'A real photo of {gender} who works as {article} {job}'

                images = pipe(prompt=prompt,
                              encoder_debias=encoder_debias,
                              decoder_debias=decoder_debias,
                              gender=gender,
                              text_important_indices_enc=text_important_indices_enc,
                              text_enc_results_dict=text_enc_results_dict,
                              image_important_indices_dec=(image_important_indices_dec_down,image_important_indices_dec_mid) ,
                              image_dec_results_dict=(image_dec_results_dict_down,image_dec_results_dict_mid),
                              )[0].images[0]
                
                # Save the image to the directory
                image_path = os.path.join(directory_path, f"{job}_{gender}_{seed}.png")
                images.save(image_path)
evaluate_t2i(directory_path,device=device)