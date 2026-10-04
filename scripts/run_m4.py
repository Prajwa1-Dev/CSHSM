import os
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizerFast
from src.data.dataset import CSHSMDataset
from src.models.multitask_xlmr import MultitaskXLMR
from src.models.trainer import MultitaskTrainer
from src.evaluation.metrics import generate_evaluation_report
import collections
import argparse

def main():
    parser = argparse.ArgumentParser(description="M4 Training Pipeline")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    print("Starting M4 Pipeline...")

    # Configuration
    config = {
        "model_name": "xlm-roberta-base",
        "batch_size": args.batch_size,
        "lr": 2e-5,
        "epochs": 5,
        "max_length": 128,
        "seed": 42,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "patience_limit": 3,
        "lambda_cat": 1.0,
        "optimizer": "AdamW",
        "scheduler": "linear_with_warmup",
        "accumulation_steps": args.accumulation_steps,
        "fp16": args.fp16
    }
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    max_length = config["max_length"]
    seed = config["seed"]
    patience_limit = config["patience_limit"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(device)}")

    # Tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

    # Check Truncation Statistics
    train_lens = []
    with open('data/splits/train.jsonl', 'r') as f:
        for line in f:
            t = json.loads(line)["Comment"]
            train_lens.append(len(tokenizer.encode(t)))
    exceeded = sum(1 for l in train_lens if l > max_length)
    pct_truncated = (exceeded / len(train_lens)) * 100
    print(f"Truncation Stats: {exceeded} examples ({pct_truncated:.2f}%) exceeded {max_length} tokens.")

    # Load Datasets
    print("Loading datasets...")
    train_dataset = CSHSMDataset('data/splits/train.jsonl', tokenizer, max_length)
    val_dataset = CSHSMDataset('data/splits/val.jsonl', tokenizer, max_length)
    test_dataset = CSHSMDataset('data/splits/test.jsonl', tokenizer, max_length)

    if args.smoke_test:
        print("SMOKE TEST MODE ENABLED.")
        epochs = 1
        config["epochs"] = 1
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(32, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(32, len(val_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(32, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Calculate class weights for categories
    cat_counts = collections.Counter([x["category_label"].item() for x in train_dataset if x["category_label"].item() != -1])
    total_cats = sum(cat_counts.values())
    cat_weights = [total_cats / cat_counts.get(i, 1) for i in range(6)]
    # Normalize weights
    cat_weights = [w / sum(cat_weights) for w in cat_weights]
    print(f"Category Weights: {cat_weights}")

    # Model & Trainer
    print("Initializing model...")
    model = MultitaskXLMR()
    trainer = MultitaskTrainer(model, train_loader, val_loader, device, category_weights=cat_weights, lr=lr, accumulation_steps=config["accumulation_steps"], use_fp16=config["fp16"])

    from transformers import get_linear_schedule_with_warmup
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    trainer.scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_score = -1
    if args.smoke_test:
        best_checkpoint = "models/smoke_test/xlmr_best.pt"
        resumable_best_checkpoint = "models/smoke_test/xlmr_resumable_best.pt"
        latest_checkpoint = "models/smoke_test/xlmr_latest.pt"
        os.makedirs("models/smoke_test", exist_ok=True)
    else:
        best_checkpoint = "models/xlmr_best.pt"
        resumable_best_checkpoint = "models/xlmr_resumable_best.pt"
        latest_checkpoint = "models/xlmr_latest.pt"
        os.makedirs("models", exist_ok=True)

    start_epoch = 0
    patience_counter = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        res_epoch, res_best_score, res_patience, res_config, res_seed = trainer.load_checkpoint(args.resume)
        start_epoch = res_epoch + 1
        best_score = res_best_score
        patience_counter = res_patience
        print(f"Resumed at epoch {res_epoch}, best validation score: {best_score:.4f}, patience: {patience_counter}")
        if start_epoch >= epochs:
            print("Training already completed.")

    start_time = time.time()

    # Training Loop
    print("Starting training...")

    for epoch in range(start_epoch, epochs):
        t_loss, b_loss, c_loss = trainer.train_epoch()
        val_metrics = trainer.evaluate(val_loader)

        score = val_metrics["combined_score"]
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} (Bin: {b_loss:.4f}, Cat: {c_loss:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Bin F1: {val_metrics['bin_f1']:.4f} | Oracle Cat F1: {val_metrics['oracle_cat_f1']:.4f} | Combined: {score:.4f}")

        if score > best_score:
            best_score = score
            # Save legacy checkpoint
            torch.save(model.state_dict(), best_checkpoint)
            # Save resumable best checkpoint
            trainer.save_checkpoint(resumable_best_checkpoint, epoch, best_score, config, seed, patience_counter=0)
            patience_counter = 0
            print("  --> Saved new best checkpoint!")
        else:
            patience_counter += 1

        # Always save latest checkpoint after epoch completion
        trainer.save_checkpoint(latest_checkpoint, epoch, best_score, config, seed, patience_counter=patience_counter)
        print("  --> Saved latest checkpoint.")

        if patience_counter >= patience_limit:
            print("Early stopping triggered.")
            break

    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes.")

    # Load Best Model and Test
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    test_metrics = trainer.evaluate(test_loader)

    final_report = generate_evaluation_report(test_metrics, "Test")

    # Error Analysis (Top false positives/negatives)
    errors = []
    bin_labels = test_metrics["bin_labels"]
    bin_preds = test_metrics["bin_preds"]
    cat_labels = test_metrics["cat_labels"]
    cat_preds = test_metrics["cat_preds"]

    test_indices = test_dataset.indices if isinstance(test_dataset, torch.utils.data.Subset) else range(len(test_dataset))
    for i, orig_idx in enumerate(test_indices):
        data = test_dataset.dataset.data[orig_idx] if isinstance(test_dataset, torch.utils.data.Subset) else test_dataset.data[i]
        if bin_labels[i] != bin_preds[i]:
            errors.append({
                "type": "Binary_Error",
                "text": data["Comment"],
                "true_label": int(bin_labels[i]),
                "pred_label": int(bin_preds[i])
            })
        elif bin_labels[i] == 1 and cat_labels[i] != -1 and cat_labels[i] != cat_preds[i]:
            errors.append({
                "type": "Category_Error",
                "text": data["Comment"],
                "true_cat": int(cat_labels[i]),
                "pred_cat": int(cat_preds[i])
            })

    report_dir = "reports/smoke_test" if args.smoke_test else "reports"
    os.makedirs(report_dir, exist_ok=True)
    with open(f"{report_dir}/m4_error_analysis.json", "w", encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    report = {
        "model_config": "xlm-roberta-base, multitask heads",
        "training_config": f"lr={lr}, batch_size={batch_size}, max_length={max_length}, epochs={epochs}",
        "training_duration_mins": train_time/60,
        "truncation_stats": {
            "exceeded_128": exceeded,
            "pct_truncated": pct_truncated
        },
        "best_val_combined_score": best_score,
        "test_results": final_report
    }

    with open(f"{report_dir}/m4_evaluation_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory Allocated: {peak_memory:.2f} MB")

    print("M4 Pipeline Finished successfully.")

if __name__ == "__main__":
    main()


