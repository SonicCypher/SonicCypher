import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def parse_result_file(file_path):
    results = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            spk, utt, _, label, _, decision = parts
            key = f"{spk}:{utt}"
            results[key] = (label, int(decision))
    print(f"Parsed {len(results)} entries from {file_path}")
    return results

def combine_results_and_decide(ASV_cosine_txt_path, prosody_cosine_txt_path):
    ASV_model = parse_result_file(ASV_cosine_txt_path)
    prosody_model = parse_result_file(prosody_cosine_txt_path)

    combined = []
    for key in ASV_model:
        if key not in prosody_model:
            continue
        label1, decision1 = ASV_model[key]
        label2, decision2 = prosody_model[key]
        final_decision = decision1 & decision2
        combined.append((label1, final_decision))
    print(f"Combined {len(combined)} entries from both models")
    return combined

def compute_metrics(predictions, label_filter=None):
    y_true, y_pred = [], []

    for label, decision in predictions:
        if label_filter and label not in label_filter:
            continue
        true_label = 1 if label == "target" else 0
        y_true.append(true_label)
        y_pred.append(decision)
    print(f"Filtered {len(y_true)} samples for labels: {label_filter}")
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0][0], cm[0][1] # True Positives – Correctly predicted target as target, False Negatives – Incorrectly predicted target as nontarget
    fp, tn = cm[1][0], cm[1][1] # False Positives – Incorrectly predicted nontarget as target, True Negatives – Correctly predicted nontarget as nontarget

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    far = fp / (fp + tn + 1e-10)
    frr = fn / (tp + fn + 1e-10)

    metrics = {
        "Accuracy": accuracy * 100,   # total predictions that were correct
        "Precision": precision * 100, # predicted targets that were actually targets
        "Recall (TPR)": recall * 100, # actual targets that were predicted as targets
        "F1 Score": f1 * 100, # harmonic mean of precision and recall
        "FAR": far * 100, # false acceptance rate : rate at which non-targets or spoofs are wrongly accepted
        "FRR": frr * 100 # false rejection rate : rate at which targets are wrongly rejected
    }

    return cm, metrics

def plot_all_dashboards(predictions, save_path=None):
    tasks = [
        ("SASV (target vs nontarget + spoof)", None),
        ("SV (target vs nontarget)", {"target", "nontarget"}),
        ("SPF (target vs spoof)", {"target", "spoof"})
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    fig.suptitle("Decision Fusion Evaluation Dashboard", fontsize=16)

    for row, (title, label_filter) in enumerate(tasks):
        cm, metrics = compute_metrics(predictions, label_filter)

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row][0])
        axes[row][0].set_title(f"{title} - Confusion Matrix")
        axes[row][0].set_xlabel("Predicted")
        axes[row][0].set_ylabel("Actual")
        if label_filter is None:
            labels_display = ["Target", "Nontarget/Spoof"]
        elif len(label_filter) == 2:
            other_label = next(label for label in label_filter if label != "target")
            labels_display = ["Target", other_label.capitalize()]
        else:
            labels_display = ["Target", "Other"]

        axes[row][0].set_xticklabels(labels_display)
        axes[row][0].set_yticklabels(labels_display)


        # Metric Bar Plot
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        axes[row][1].barh(metric_names, metric_values, color="teal")
        axes[row][1].set_xlim(0, 100)
        axes[row][1].set_title(f"{title} - Metrics (%)")
        for i, v in enumerate(metric_values):
            axes[row][1].text(v + 1, i, f"{v:.2f}%", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Paths to the model outputs
    ASV_cosine_txt_path = "/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/cosine_scores_eval.txt"
    prosody_cosine_txt_path = "/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/prosody_model_results.txt"

    
    combined_predictions = combine_results_and_decide(ASV_cosine_txt_path, prosody_cosine_txt_path)
    # 🔍 Inspect the labels to debug filtering issues
    unique_labels = set(label for label, _ in combined_predictions)
    print("Unique labels found:", unique_labels)

    # Path to save the output image
    image_folder = "/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/images"
    os.makedirs(image_folder, exist_ok=True)  # Create folder if it doesn’t exist
    # Save path for the output image
    output_image_path = os.path.join(image_folder, "decision_fusion_dashboard.png")

    # Show unified dashboard
    plot_all_dashboards(combined_predictions, save_path=output_image_path)
