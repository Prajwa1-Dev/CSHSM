import os
import json
import datetime
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.multitask_xlmr import MultitaskXLMR
from src.models.trainer import MultitaskTrainer
from src.data.dataset import CSHSMDataset
from transformers import XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
import emoji

def get_language_profile(text):
    kannada_chars = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z' or 'A' <= c <= 'Z')
    
    if kannada_chars > english_chars * 2:
        return "Kannada-heavy"
    elif english_chars > kannada_chars * 2:
        return "English-heavy"
    else:
        return "Code-mixed"

def main():
    # 1. Generate Experiment Record
    os.makedirs("experiments", exist_ok=True)
    
    try:
        chk = torch.load("models/xlmr_resumable_best.pt", map_location="cpu", weights_only=False)
        config = chk.get("config", {})
    except Exception as e:
        print(f"Could not load resumable checkpoint: {e}")
        config = {}

    try:
        with open("reports/m4_evaluation_report.json", "r", encoding="utf-8") as f:
            eval_report = json.load(f)
    except Exception as e:
        print(f"Could not load eval report: {e}")
        eval_report = {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_record = {
        "timestamp": timestamp,
        "model_name": config.get("model_name", "xlm-roberta-base"),
        "architecture_description": "Shared XLM-R backbone with Binary and 6-class Category heads",
        "learning_rate": config.get("lr", "Unavailable"),
        "batch_size": config.get("batch_size", "Unavailable"),
        "accumulation_steps": config.get("accumulation_steps", "Unavailable"),
        "max_sequence_length": config.get("max_length", "Unavailable"),
        "epochs": config.get("epochs", "Unavailable"),
        "seed": config.get("seed", "Unavailable"),
        "weight_decay": config.get("weight_decay", "Unavailable"),
        "warmup_ratio": config.get("warmup_ratio", "Unavailable"),
        "lambda_cat": config.get("lambda_cat", "Unavailable"),
        "optimizer": config.get("optimizer", "Unavailable"),
        "scheduler": config.get("scheduler", "Unavailable"),
        "fp16": config.get("fp16", "Unavailable"),
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "training_duration_mins": eval_report.get("training_duration_mins", "Unavailable"),
        "truncation_stats": eval_report.get("truncation_stats", "Unavailable"),
        "best_val_combined_score": eval_report.get("best_val_combined_score", "Unavailable"),
        "test_binary_metrics": eval_report.get("test_results", {}).get("binary_report", "Unavailable"),
        "test_oracle_category_metrics": eval_report.get("test_results", {}).get("oracle_report", "Unavailable"),
        "test_cascaded_category_metrics": eval_report.get("test_results", {}).get("cascaded_report", "Unavailable"),
        "checkpoint_used_for_evaluation": "models/xlmr_best.pt"
    }

    exp_path = f"experiments/exp_{timestamp}.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"Saved experiment record to {exp_path}")

    # 2. Aggregate Error Analysis
    print("Running aggregate error analysis on held-out test set...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config.get("model_name", "xlm-roberta-base"))
    test_dataset = CSHSMDataset('data/splits/test.jsonl', tokenizer, config.get("max_length", 128))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskXLMR(model_name=config.get("model_name", "xlm-roberta-base"))
    model.load_state_dict(torch.load("models/xlmr_best.pt", map_location=device, weights_only=True))
    
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 16))
    trainer = MultitaskTrainer(model, test_loader, test_loader, device, use_fp16=config.get("fp16", False))
    
    metrics = trainer.evaluate(test_loader)
    
    bin_preds = metrics["bin_preds"]
    bin_labels = metrics["bin_labels"]
    cat_preds = metrics["cat_preds"]
    cat_labels = metrics["cat_labels"]
    
    categories = ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"]
    
    analysis = {
        "overall_total": len(test_dataset),
        "overall_errors": 0,
        "by_category": {cat: {"total": 0, "errors": 0} for cat in categories},
        "by_language_profile": {"Kannada-heavy": {"total": 0, "errors": 0}, "English-heavy": {"total": 0, "errors": 0}, "Code-mixed": {"total": 0, "errors": 0}},
        "by_text_length": {"Short (<15 words)": {"total": 0, "errors": 0}, "Long (>=15 words)": {"total": 0, "errors": 0}},
        "by_emoji": {"Contains Emoji": {"total": 0, "errors": 0}, "No Emoji": {"total": 0, "errors": 0}},
        "qualitative_errors": []
    }
    
    for i in range(len(test_dataset)):
        data = test_dataset.data[i]
        text = data["Comment"]
        
        is_bin_error = (bin_preds[i] != bin_labels[i])
        is_cat_error = (bin_labels[i] == 1 and cat_labels[i] != -1 and cat_preds[i] != cat_labels[i])
        is_error = is_bin_error or is_cat_error
        
        if is_error:
            analysis["overall_errors"] += 1
            if is_bin_error:
                analysis["qualitative_errors"].append({
                    "type": "Binary_Error",
                    "text": text,
                    "true_label": int(bin_labels[i]),
                    "pred_label": int(bin_preds[i])
                })
            elif is_cat_error:
                analysis["qualitative_errors"].append({
                    "type": "Category_Error",
                    "text": text,
                    "true_cat": int(cat_labels[i]),
                    "pred_cat": int(cat_preds[i])
                })

        true_cat = cat_labels[i]
        if true_cat != -1 and true_cat < len(categories):
            cat_name = categories[true_cat]
            analysis["by_category"][cat_name]["total"] += 1
            if is_error:
                analysis["by_category"][cat_name]["errors"] += 1
                
        lang_prof = get_language_profile(text)
        analysis["by_language_profile"][lang_prof]["total"] += 1
        if is_error:
            analysis["by_language_profile"][lang_prof]["errors"] += 1
            
        words = len(text.split())
        len_key = "Short (<15 words)" if words < 15 else "Long (>=15 words)"
        analysis["by_text_length"][len_key]["total"] += 1
        if is_error:
            analysis["by_text_length"][len_key]["errors"] += 1
            
        has_emoji = emoji.emoji_count(text) > 0
        emo_key = "Contains Emoji" if has_emoji else "No Emoji"
        analysis["by_emoji"][emo_key]["total"] += 1
        if is_error:
            analysis["by_emoji"][emo_key]["errors"] += 1

    for key, subdict in analysis.items():
        if key not in ["by_category", "by_language_profile", "by_text_length", "by_emoji"]:
            continue
        for subkey, stats in subdict.items():
            if stats["total"] > 0:
                stats["error_rate"] = stats["errors"] / stats["total"]
            else:
                stats["error_rate"] = 0.0

    analysis_path = "reports/m4_aggregate_error_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
        
    print(f"Saved aggregate error analysis to {analysis_path}")

if __name__ == '__main__':
    main()
