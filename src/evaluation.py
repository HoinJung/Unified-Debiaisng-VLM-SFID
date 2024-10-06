import torch
import clip
from utils import FacetDataset, evaluate_recall, evaluate_gender_difference, zero_shot_classifier, calculate_accuracy
from torch.utils.data import DataLoader
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from pycocoevalcap.spice.spice import Spice
import ast
import re
import os
from pprint import pprint
import subprocess
import threading
from sklearn.utils import resample

from pycocoevalcap.meteor import meteor as meteor_module  # Import the module

# Locate the directory where the 'meteor.py' is located
meteor_dir = os.path.dirname(os.path.abspath(meteor_module.__file__))
# Construct the full path to 'meteor-1.5.jar'
METEOR_JAR = os.path.join(meteor_dir, 'meteor-1.5.jar')
class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                # cwd=os.path.dirname(os.path.abspath(__file__)), \
                cwd = meteor_dir,\
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0,len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = score_line.replace('\n', '').replace('\r', '')
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = score_line.replace('\n', '').replace('\r', '')
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()

def evaluate_flickr(args, clip_model, device,  clip_name, img_important_indices=None, img_mean_features_lowconfidence=None, text_important_indices=None, text_mean_features_lowconfidence=None):
    model, preprocess = clip.load(clip_model, device=device)

    batch_size = 512
    all_text_features = []
    all_image_features = []
    all_results = []

    for version in tqdm(range(0,10)):
        text_path = f'embedding/flickr1000_{clip_name}_text_v{version}.pt'
        image_path = f'embedding/flickr1000_{clip_name}_image_v{version}.pt'
        text_embeddings = torch.load(text_path)
        image_embeddings = torch.load(image_path)
        text_features = text_embeddings['text_embeddings'].float()
        image_features = image_embeddings['image_embeddings'].float()
        all_text_features.append(text_features)
        all_image_features.append(image_features)
        text_embedding = text_features
        image_embedding = image_features
        image_id = image_embeddings['id']
        text_id = text_embeddings['id']
        img_gender = image_embeddings['gender']
        text_gender = text_embeddings['gender']

        if 'image' in args.target:
            image_embedding = image_embedding.to(device)
            image_embedding[:, img_important_indices] = img_mean_features_lowconfidence[img_important_indices]
            image_embedding = image_embedding.cpu()
            
        if 'text' in args.target:
            text_embedding = text_embedding.to(device)
            text_embedding[:, text_important_indices] = text_mean_features_lowconfidence[text_important_indices]
            text_embedding = text_embedding.cpu()
        return_results = []
        for top_k in [1, 5, 10]:
            recall_results = evaluate_recall(image_embedding, image_id, text_embedding, text_id, top_k)
            # print(f'Seed {version} {clip_name}: Recall@{top_k}:', recall_results * 100)
            return_results.append(recall_results)
        for top_k in [100]:
            gender_difference = evaluate_gender_difference(image_embedding, image_id, img_gender, text_embedding, text_id, text_gender, top_k)
            # print(f"Seed {version} {clip_name}: Top-{top_k} Gender Difference in Retrieved Images: MaxSkew {gender_difference}")
            return_results.append(gender_difference)
        all_results.append(return_results)

    all_results = np.array(all_results)
    means = np.mean(all_results, axis=0)
    stds = np.std(all_results, axis=0)
    means = np.round(means, 4)
    stds = np.round(stds, 4)
    # print(f'Mean results: {means}')
    # print(f'Standard deviation of results: {stds}')
    mean_labels = ['Recall@1', 'Recall@5', 'Recall@10', 'Skew']

    for i in range(len(means)):
        print(f'{mean_labels[i]}: {means[i]:.4f} +/- {stds[i]:.4f}')


    return tuple(means), tuple(stds)
def get_text_embeddings(args,clip_model, device, prompts, 
                        text_important_indices, text_mean_features_lowconfidence):
    model, preprocess = clip.load(clip_model, device=device)
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        if args.mode == 'sfid':
            text_embedding = text_embedding.float()
            text_embedding[:, text_important_indices] = text_mean_features_lowconfidence[text_important_indices]
            # text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding_norm
    
def get_image_embeddings(val_dataloader, args, device, 
                         img_important_indices, img_mean_features_lowconfidence):
    image_embeddings_all = []
    class_labels_all = []
    genders_all = []

    for data in val_dataloader:
        image_embeddings, genders, _, _, class_labels = data
        if 'image' in args.target:
            if args.mode  =='sfid':
                image_embeddings = image_embeddings.to(device)
                image_embeddings[:, img_important_indices] = img_mean_features_lowconfidence[img_important_indices]
                # image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings_all.append(image_embeddings)
        class_labels_all.append(class_labels)
        genders_all.append(genders)

    return torch.cat(image_embeddings_all), torch.cat(class_labels_all), torch.cat(genders_all)

def evaluate_facet(args, clip_model, device,  clip_name,
                   img_important_indices=None, img_mean_features_lowconfidence=None,
                   text_important_indices=None, text_mean_features_lowconfidence=None):
    annotations = pd.read_csv('../data/facet/annotations/annotations.csv')
    class_list = annotations['class1'].unique().tolist()
    unique_class_names = sorted(set(class_list))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_class_names)}
    article_list = ['an' if job[0] in ['a','e','i','o','u'] else 'a' for job in unique_class_names ]
    prompts = [f'a photo of {article} {job}' for article, job in zip(article_list, unique_class_names)]

    val_dataset = FacetDataset(clip_name=clip_name, device=device, class_to_idx=class_to_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    text_embeddings = get_text_embeddings(args,clip_model, device, prompts,
                                          text_important_indices, text_mean_features_lowconfidence)

    image_embeddings_all, class_labels_all, genders_all = get_image_embeddings(val_dataloader, args, device, img_important_indices,
                                                                               img_mean_features_lowconfidence)
    print('Run Bootstrapping for FACET dataset.')
    def bootstrap_evaluation():
        bootstrap_accuracies = []
        bootstrap_mean_dps = []
        bootstrap_max_dps = []

        dataset_size = len(class_labels_all)
        indices = list(range(dataset_size))

        for _ in tqdm(range(100)):
            sampled_indices = random.choices(indices, k=dataset_size)
            sampled_image_embeddings = image_embeddings_all[sampled_indices]
            sampled_class_labels = class_labels_all[sampled_indices]
            sampled_genders = genders_all[sampled_indices]

            total_accuracy = 0
            total_accuracy_count = 0
            total_accuracy_difference_sum = 0
            total_accuracy_difference_count = 0
            predictions_by_class_gender = defaultdict(lambda: defaultdict(list))
            true_labels_by_class_gender = defaultdict(lambda: defaultdict(list))
            class_gender_counts = defaultdict(lambda: defaultdict(int))
            class_gender_totals = defaultdict(lambda: defaultdict(int))

            predicted_labels = zero_shot_classifier(sampled_image_embeddings, text_embeddings, unique_class_names)

            for pred, true, gender in zip(predicted_labels, sampled_class_labels.cpu().numpy(), sampled_genders.cpu().numpy()):
                pred = class_to_idx[pred]
                gender_key = 'Male' if gender == 1 else 'Female'
                class_key = unique_class_names[true]
                predictions_by_class_gender[class_key][gender_key].append(pred)
                true_labels_by_class_gender[class_key][gender_key].append(true)
                class_gender_counts[class_key][gender_key] += (pred == true)
                class_gender_totals[class_key][gender_key] += 1

            accuracy_by_class_gender = defaultdict(dict)
            for class_key, genders in true_labels_by_class_gender.items():
                for gender_key in genders.keys():
                    accuracy = calculate_accuracy(predictions_by_class_gender[class_key][gender_key],
                                                  true_labels_by_class_gender[class_key][gender_key])
                    accuracy_by_class_gender[class_key][gender_key] = accuracy
                    total_accuracy += accuracy
                    total_accuracy_count += 1

            for class_key, gender_accuracies in accuracy_by_class_gender.items():
                if 'Male' in gender_accuracies and 'Female' in gender_accuracies:
                    accuracy_difference = abs(gender_accuracies['Male'] - gender_accuracies['Female'])
                    total_accuracy_difference_sum += accuracy_difference
                    total_accuracy_difference_count += 1

            demographic_parities = []
            for class_key, counts in class_gender_counts.items():
                if 'Male' in counts and 'Female' in counts:
                    P_yk_given_a0 = counts['Female'] / class_gender_totals[class_key]['Female'] if class_gender_totals[class_key]['Female'] > 0 else 0
                    P_yk_given_a1 = counts['Male'] / class_gender_totals[class_key]['Male'] if class_gender_totals[class_key]['Male'] > 0 else 0
                    demographic_parity = abs(P_yk_given_a0 - P_yk_given_a1) * 100
                    demographic_parities.append(demographic_parity)

            average_accuracy = (total_accuracy / total_accuracy_count) if total_accuracy_count else 0
            mean_demographic_parity = np.mean(demographic_parities) if demographic_parities else 0
            max_demographic_parity = max(demographic_parities) if demographic_parities else 0

            bootstrap_accuracies.append(average_accuracy)
            bootstrap_mean_dps.append(mean_demographic_parity)
            bootstrap_max_dps.append(max_demographic_parity)

        return np.mean(bootstrap_accuracies), np.std(bootstrap_accuracies), \
               np.mean(bootstrap_mean_dps), np.std(bootstrap_mean_dps), \
               np.mean(bootstrap_max_dps), np.std(bootstrap_max_dps)
    mean_acc, std_acc, mean_dp, std_dp, max_dp, std_max_dp = bootstrap_evaluation()
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    print(f"Mean Demographic Parity: {mean_dp:.2f} (+/- {std_dp:.2f})")
    print(f"Max Demographic Parity: {max_dp:.2f} (+/- {std_max_dp:.2f})")

    return mean_acc, std_acc, mean_dp, std_dp, max_dp, std_max_dp



def misclassification_rate(df):
    total_males = df[df['ground_truth_gender'] == 'Male'].shape[0]
    total_females = df[df['ground_truth_gender'] == 'Female'].shape[0]
    
    male_lowconfidence = df[(df['ground_truth_gender'] == 'Male') & (df['detected_gender'] == 'Female')].shape[0]
    female_lowconfidence = df[(df['ground_truth_gender'] == 'Female') & (df['detected_gender'] == 'Male')].shape[0]
    
    male_misclassification_rate = male_lowconfidence / total_males if total_males > 0 else 0
    female_misclassification_rate = female_lowconfidence / total_females if total_females > 0 else 0
    
    overall_misclassification_rate = (male_lowconfidence + female_lowconfidence) / (total_males + total_females)
    
    # Calculate Composite Misclassification Rate (MR_C)
    composite_misclassification_rate = np.sqrt(
        overall_misclassification_rate**2 + (female_misclassification_rate - male_misclassification_rate)**2
    )
    
    return {
        'Male Misclassification Rate': round(male_misclassification_rate*100, 2),
        'Female Misclassification Rate': round(female_misclassification_rate*100, 2),
        'Overall Misclassification Rate': round(overall_misclassification_rate*100, 2),
        'Composite Misclassification Rate': round(composite_misclassification_rate*100, 2)
    }


# Function to convert string list representations to actual lists
def convert_str_to_list(str_list):
    try:
        return ast.literal_eval(str_list)
    except ValueError:
        return []  # Returns an empty list in case of error


def neutralize_gender(text):
    """ Neutralize gendered words in the given text. """
    gendered_words = {
        r"\bman\b": "person", r"\bguy\b": "person", r"\bson\b": "child", r"\bboy\b": "child",
        r"\bwoman\b": "person", r"\blady\b": "person",
        r"\bmen\b": "people", r"\bwomen\b": "people"
    }
    for word, neutral in gendered_words.items():
        text = re.sub(word, neutral, text, flags=re.IGNORECASE)
    return text


def evaluate_captions_max(df):
    """ Evaluate captions taking the maximum score between original and neutralized ground truths. """
    gts = {}
    res = {}
    gts_neutral = {}
    for i, row in df.iterrows():
        original_gts = row['gt_captions']
        gts[i] = original_gts
        res[i] = [row['generated_text']]
        # Neutralize each caption in the ground truths
        gts_neutral[i] = [neutralize_gender(caption) for caption in original_gts]
        
    scorers = [
        (Meteor(), ["METEOR"]),
        (Spice(), ["SPICE"])
    ]

    results = {method[0]: 0 for scorer, method in scorers}

    for scorer, method in tqdm(scorers):
        score_orig, scores_orig = scorer.compute_score(gts, res)
        score_neutral, scores_neutral = scorer.compute_score(gts_neutral, res)
        
        if method[0] == "SPICE":
            # Extract the F1 scores from the SPICE results
            f_scores_orig = [score['All']['f'] for score in scores_orig]
            f_scores_neutral = [score['All']['f'] for score in scores_neutral]
            max_scores = [max(orig, neut) for orig, neut in zip(f_scores_orig, f_scores_neutral)]
        else:
            max_scores = [max(orig, neut) for orig, neut in zip(scores_orig, scores_neutral)]
        results[method[0]] = sum(max_scores) / len(max_scores)
    return results


def report_df(df):
    df['gt_captions'] = df['gt_captions'].apply(convert_str_to_list)
    rates = misclassification_rate(df)
    results = evaluate_captions_max(df)

    return rates, results

def bootstrap(df, num_samples=100, sample_size=1000):
    bootstrap_results = []
    for _ in tqdm(range(num_samples)):
        sample_df = resample(df, n_samples=sample_size)
        rates, results = report_df(sample_df)
        bootstrap_results.append((rates, results))
    return bootstrap_results

def calculate_confidence_intervals(bootstrap_results, confidence_level=0.95):
    metrics = ['Male Misclassification Rate', 'Female Misclassification Rate', 'Overall Misclassification Rate',
               'Composite Misclassification Rate', 'METEOR', 'SPICE']
    ci_lower = {}
    ci_upper = {}
    for metric in metrics:
        values = [result[0][metric] if metric in result[0] else result[1][metric] for result in bootstrap_results]
        
        lower_bound = np.percentile(values, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(values, (1 + confidence_level) / 2 * 100)
        ci_lower[metric] = lower_bound
        ci_upper[metric] = upper_bound
        
    return ci_lower, ci_upper


def evaluate_image_captioning(file_path):
    columns = ['File', 'Male Misclassification Rate', 'Female Misclassification Rate',
               'Overall Misclassification Rate', 'Composite Misclassification Rate', 'METEOR', 'SPICE']

    print(f'Evaluating Image Captioning for {file_path}')
    df = pd.read_csv(file_path)
    
    # Run bootstrapping and calculate confidence intervals
    bootstrap_results = bootstrap(df)
    ci_lower, ci_upper = calculate_confidence_intervals(bootstrap_results)
    
    # Function to calculate mean and margin
    def mean_margin(lower, upper):
        mean = (lower + upper) / 2
        margin = (upper - lower) / 2
        return mean, margin
    
    # Calculate mean and margin for each metric
    male_mis_mean, male_mis_margin = mean_margin(ci_lower['Male Misclassification Rate'], ci_upper['Male Misclassification Rate'])
    female_mis_mean, female_mis_margin = mean_margin(ci_lower['Female Misclassification Rate'], ci_upper['Female Misclassification Rate'])
    overall_mis_mean, overall_mis_margin = mean_margin(ci_lower['Overall Misclassification Rate'], ci_upper['Overall Misclassification Rate'])
    composite_mis_mean, composite_mis_margin = mean_margin(ci_lower['Composite Misclassification Rate'], ci_upper['Composite Misclassification Rate'])
    meteor_mean, meteor_margin = mean_margin(ci_lower['METEOR']*100, ci_upper['METEOR']*100)
    spice_mean, spice_margin = mean_margin(ci_lower['SPICE']*100, ci_upper['SPICE']*100)

    # Prepare the result row with mean ± margin format
    new_row = {
        'file_path': file_path,
        'Male Misclassification Rate': f"{male_mis_mean:.2f} ± {male_mis_margin:.2f}",
        'Female Misclassification Rate': f"{female_mis_mean:.2f} ± {female_mis_margin:.2f}",
        'Overall Misclassification Rate': f"{overall_mis_mean:.2f} ± {overall_mis_margin:.2f}",
        'Composite Misclassification Rate': f"{composite_mis_mean:.2f} ± {composite_mis_margin:.2f}",
        'METEOR': f"{meteor_mean:.2f} ± {meteor_margin:.2f}",
        'SPICE': f"{spice_mean:.2f} ± {spice_margin:.2f}"
    }

    # Print the result in the terminal
    pprint(new_row)
    # columns = ['File', 'Male Misclassification Rate', 'Female Misclassification Rate',
    #            'Overall Misclassification Rate', 'Composite Misclassification Rate', 'METEOR', 'SPICE']

    # print(f'Evaluate Image Captioning for {file_path}')
    # df = pd.read_csv(file_path)
    
    # bootstrap_results = bootstrap(df)
    # ci_lower, ci_upper = calculate_confidence_intervals(bootstrap_results)

    # new_row = {
    #     'file_path': file_path,
    #     'Male Misclassification Rate': f"{ci_lower['Male Misclassification Rate']:.2f}-{ci_upper['Male Misclassification Rate']:.2f}",
    #     'Female Misclassification Rate': f"{ci_lower['Female Misclassification Rate']:.2f}-{ci_upper['Female Misclassification Rate']:.2f}",
    #     'Overall Misclassification Rate': f"{ci_lower['Overall Misclassification Rate']:.2f}-{ci_upper['Overall Misclassification Rate']:.2f}",
    #     'Composite Misclassification Rate': f"{ci_lower['Composite Misclassification Rate']:.2f}-{ci_upper['Composite Misclassification Rate']:.2f}",
    #     'METEOR': f"{ci_lower['METEOR']*100:.2f}-{ci_upper['METEOR']*100:.2f}",
    #     'SPICE': f"{ci_lower['SPICE']*100:.2f}-{ci_upper['SPICE']*100:.2f}"
    # }
    # pprint(new_row)
    