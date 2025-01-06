import os
import torch
import numpy as np
from sklearn.metrics import roc_curve

# 1. **Load Model**
def load_model(model_path, device="cpu"):
    """Loads the model from a specified path."""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("path_to/best_model.pth", device=device)


# 2. **Load Embeddings**
def load_embeddings_from_npy(folder_path):
    """Loads embeddings from .npy files in a folder."""
    embeddings = {}
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):  # Check for .npy files
            file_path = os.path.join(folder_path, file)
            emb_id = os.path.splitext(file)[0]  # Get ID from filename
            embeddings[emb_id] = torch.tensor(np.load(file_path), dtype=torch.float32)
    return embeddings

# Example usage for loading embeddings
enrol_dict = load_embeddings_from_npy("path_to/enrol_folder")
test_dict = load_embeddings_from_npy("path_to/test_folder")
train_dict = load_embeddings_from_npy("path_to/train_folder") if "score_norm" in params else None


# 3. **Load Verification List**
def load_verification_list(file_path):
    """Loads the verification list from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        veri_test = [line.strip() for line in f]
    return veri_test

# Example usage
veri_test = load_verification_list("path_to/verification_file.txt")


# 4. **Cosine Similarity & Score Normalization**
def get_verification_scores(veri_test, enrol_dict, test_dict, train_dict=None, params=None):
    """Computes positive and negative scores given the verification split."""
    scores = []
    positive_scores = []
    negative_scores = []

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Optional score normalization
    if params and "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    for line in veri_test:
        lab_pair, enrol_id, test_id = line.split()
        lab_pair = int(lab_pair)

        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        if params and "score_norm" in params:
            # Normalize using train embeddings
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)

            score_e_c = similarity(enrol_rep, train_cohort)
            score_t_c = similarity(test_rep, train_cohort)

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute cosine similarity
        score = similarity(enrol, test)[0]

        if params and "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # Append scores
        scores.append(score.item())
        if lab_pair == 1:
            positive_scores.append(score.item())
        else:
            negative_scores.append(score.item())

    return positive_scores, negative_scores


# 5. **EER Calculation (provided)**
def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold)."""
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    intermediate_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, intermediate_thresholds]))

    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]

        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    EER = (final_FAR + final_FRR) / 2
    return float(EER), float(thresholds[min_index])


# 6. **minDCF Calculation (provided)**
def minDCF(positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems."""
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    intermediate_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, intermediate_thresholds]))

    positive_scores = torch.cat(len(thresholds) * [positive_scores.unsqueeze(0)])
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]

    negative_scores = torch.cat(len(thresholds) * [negative_scores.unsqueeze(0)])
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])


# 7. **Running the Full Pipeline**
def main(params):
    # Load verification list
    veri_test = load_verification_list("path_to/verification_file.txt")

    # Compute verification scores
    positive_scores, negative_scores = get_verification_scores(
        veri_test, enrol_dict, test_dict, train_dict, params=params
    )

    # Calculate EER and minDCF
    eer, threshold_eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    print(f"EER: {eer * 100:.2f}%")

    min_dcf, threshold_dcf = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))
    print(f"minDCF: {min_dcf * 100:.2f}%")


if __name__ == "__main__":
    # Assuming params are loaded from your YAML/config
    params = {
        "score_norm": "z-norm",  # Example score normalization setting
        # Add other parameters as necessary
    }

    # Run the main pipeline
    main(params)
