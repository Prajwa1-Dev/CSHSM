from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np

def generate_evaluation_report(metrics, dataset_name="Test"):
    bin_labels = metrics["bin_labels"]
    bin_preds = metrics["bin_preds"]
    cat_labels = np.array(metrics["cat_labels"])
    cat_preds = np.array(metrics["cat_preds"])
    
    # Binary
    bin_report = classification_report(bin_labels, bin_preds, target_names=["Non-Hate", "Hate"], output_dict=True)
    
    # Oracle Category
    valid_mask = cat_labels != -1
    oracle_labels = cat_labels[valid_mask]
    oracle_preds = cat_preds[valid_mask]
    cat_names = ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"]
    
    # Filter names to those present to avoid warnings
    present_classes = sorted(list(set(oracle_labels)))
    present_names = [cat_names[i] for i in present_classes]
    
    oracle_report = classification_report(oracle_labels, oracle_preds, labels=present_classes, target_names=present_names, output_dict=True, zero_division=0)
    
    # Cascaded Category
    # If bin_pred is Non-Hate (0), it short-circuits. So it implicitly predicts "No Category"
    # But we only evaluate category accuracy for TRUE hate examples.
    # If it's a true hate example but predicted as non-hate, it's a category miss.
    # We can represent a missed category as predicting a special 'Missed' class (e.g. -1),
    # but standard is to penalize recall for the true class.
    cascaded_preds = []
    for i in range(len(cat_preds)):
        if bin_preds[i] == 0:
            cascaded_preds.append(-1) # Short-circuited
        else:
            cascaded_preds.append(cat_preds[i])
            
    cascaded_preds = np.array(cascaded_preds)[valid_mask]
    # To compute macro F1 for the 6 categories, we evaluate how well the model predicted 0-5.
    cascaded_report = classification_report(oracle_labels, cascaded_preds, labels=present_classes, target_names=present_names, output_dict=True, zero_division=0)
    
    return {
        "binary_macro_f1": bin_report["macro avg"]["f1-score"],
        "oracle_category_macro_f1": oracle_report["macro avg"]["f1-score"] if "macro avg" in oracle_report else 0.0,
        "cascaded_category_macro_f1": cascaded_report["macro avg"]["f1-score"] if "macro avg" in cascaded_report else 0.0,
        "binary_report": bin_report,
        "oracle_report": oracle_report,
        "cascaded_report": cascaded_report
    }
