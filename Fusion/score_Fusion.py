import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from collections import Counter
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
import logging
import joblib
import os 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Step 1: Load score files
def load_score_file(file_path):
    entries = []
    logging.info(f"Loading scores from {file_path}")
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Reading {file_path}"):
            parts = line.strip().split()
            if len(parts) == 5:
                spk_id, utt_id, _, label, score = parts
                entries.append((label, utt_id, float(score)))
    return entries

# Step 2: Align scores based on utt_id
def align_entries(entries1, entries2):
    logging.info("Aligning entries...")
    score_dict1 = {utt_id: (label, score) for label, utt_id, score in entries1}
    score_dict2 = {utt_id: (label, score) for label, utt_id, score in entries2}
    
    aligned = []
    for utt_id in tqdm(score_dict1, desc="Aligning by utt_id"):
        if utt_id in score_dict2:
            label1, score1 = score_dict1[utt_id]
            label2, score2 = score_dict2[utt_id]
            if label1 == label2:
                aligned.append((label1, utt_id, score1, score2))
            else:
                logging.warning(f"Label mismatch for {utt_id}: {label1} vs {label2}")
    logging.info(f"Total aligned entries: {len(aligned)}")
    logging.info(f"Entries1 count: {len(entries1)}")
    logging.info(f"Entries2 count: {len(entries2)}")
    return aligned

def plot_score_distributions(X_raw, X_scaled, identifier=""):
    X_raw = np.array(X_raw)
    X_scaled = np.array(X_scaled)

    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    titles = ["ASV Raw", "CM Raw", "ASV Scaled", "CM Scaled"]

    for i in range(2):
        axes[0, i].hist(X_raw[:, i], bins=50, alpha=0.7, color='blue')
        axes[0, i].set_title(titles[i])
        axes[0, i].grid(True)

        axes[1, i].hist(X_scaled[:, i], bins=50, alpha=0.7, color='green')
        axes[1, i].set_title(titles[i + 2])
        axes[1, i].grid(True)

    plt.tight_layout()
    filename = f"images/score_distributions_{identifier}.png" if identifier else f"images/score_distributions.png"
    plt.savefig(filename)
    plt.show()

# Step 3: Train fusion model on dev set
def train_fusion_model(aligned_dev_entries):
    logging.info("Training fusion model with hyperparameter tuning...")
    X, y = [], []
    for label, utt_id, score1, score2 in tqdm(aligned_dev_entries, desc="Preparing training data"):
        X.append([score1, score2])
        y.append(1 if label == "target" else 0)

    logging.info(f"Training label counts: {Counter(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Visualize score distributions before and after normalization
    plot_score_distributions(X, X_scaled,identifier="train")
    
    # Define parameter grids for hyperparameter tuning
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced']
        },
        'SVM': {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(10, 5), (50, 30), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }

    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Neural Network': MLPClassifier()
    }

    best_model = None
    best_score = -1
    best_params = None

    # Perform GridSearchCV for each model
    for model_name, model in models.items():
        logging.info(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_scaled, y)

        score = grid_search.best_score_
        params = grid_search.best_params_

        logging.info(f"Best Score for {model_name}: {score:.3f}")
        logging.info(f"Best Parameters for {model_name}: {params}")

        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_params = params

    print("\n")
    logging.info(f"Best Model: {best_model}")  
    logging.info(f"Best Hyperparameters: {best_params}")
    logging.info("Fusion model trained with best hyperparameters.")

    # SAVE the trained model and scaler
    joblib.dump(best_model, "fusion_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    logging.info("Saved fusion_model.pkl and scaler.pkl")
    return best_model, scaler

# Step 4: Apply fusion model to eval set
def apply_fusion_model(clf, scaler, aligned_eval_entries):
    logging.info("Applying fusion model to eval set...")
    fused_scores = []
    X = []
    utt_info = []

    for label, utt_id, score1, score2 in tqdm(aligned_eval_entries, desc="Preparing eval features"):
        X.append([score1, score2])
        utt_info.append((label, utt_id))

    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]  # Probability of class "target"

    # Visualize score distributions before and after normalization
    plot_score_distributions(X, X_scaled, identifier="eval")

    for i, score in enumerate(probs):
        label, utt_id = utt_info[i]
        fused_scores.append((label, utt_id, score))

    logging.info("Fused scores computed.")
    return fused_scores

# Step 5: Compute EER for a given trial set
def compute_eer(scores, target_labels):
    y_true = [1 if label in target_labels else 0 for label, _, _ in scores]
    y_scores = [score for _, _, score in scores]

    if len(set(y_true)) < 2:
        logging.warning("Not enough label variety for EER calculation.")
        return float('nan'), float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fnr - fpr)
    eer_index = np.nanargmin(abs_diffs)
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold

# Step 6: Compute all EERs
def compute_all_eers(fused_scores):
    logging.info("Computing EERs...")
    sasv_eer, sasv_threshold= compute_eer(fused_scores, ["target"])
    sv_scores = [x for x in fused_scores if x[0] in ["target", "nontarget"]]
    spf_scores = [x for x in fused_scores if x[0] in ["target", "spoof"]]

    sv_eer, sv_threshold = compute_eer(sv_scores, ["target"])
    spf_eer, spf_threshold = compute_eer(spf_scores, ["target"])
    return sasv_eer, sv_eer, spf_eer, sasv_threshold, sv_threshold, spf_threshold


# Example usage:
ASV_dev_file = "Fusion/output/cosine_scores_dev.txt"
prosody_dev_file = "Fusion/output/eval_scores_using_best_dev_model_dev_score.txt"
ASV_eval_file = "Fusion/output/cosine_scores_eval.txt"
prosody_eval_file = "Fusion/output/eval_scores_using_best_dev_model_eval_score.txt"

# Load
ASV_dev_entries = load_score_file(ASV_dev_file)
prosody_dev_entries = load_score_file(prosody_dev_file)
ASV_eval_entries = load_score_file(ASV_eval_file)
prosody_eval_entries = load_score_file(prosody_eval_file)

# Align
aligned_dev = align_entries(ASV_dev_entries, prosody_dev_entries)
aligned_eval = align_entries(ASV_eval_entries, prosody_eval_entries)

# Train and Apply
fusion_model, scaler = train_fusion_model(aligned_dev)
fused_eval_scores = apply_fusion_model(fusion_model, scaler, aligned_eval)

# EERs
sasv_eer, sv_eer, spf_eer, sasv_threshold, sv_threshold, spf_threshold= compute_all_eers(fused_eval_scores)

print(f"SASV EER: {sasv_eer*100:.2f}% (Threshold: {sasv_threshold:})")
print(f"SV   EER: {sv_eer*100:.2f}% (Threshold: {sv_threshold:})")
print(f"SPF  EER: {spf_eer*100:.2f}% (Threshold: {spf_threshold:})")