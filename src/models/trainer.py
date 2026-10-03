import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import time

class MultitaskTrainer:
    def __init__(self, model, train_loader, val_loader, device, category_weights=None, lr=2e-5, lambda_cat=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_cat = lambda_cat

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        self.binary_criterion = nn.CrossEntropyLoss()
        if category_weights is not None:
            self.category_criterion = nn.CrossEntropyLoss(weight=torch.tensor(category_weights, dtype=torch.float).to(device), ignore_index=-1)
        else:
            self.category_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def train_epoch(self):
        self.model.train()
        total_loss, total_bin_loss, total_cat_loss = 0, 0, 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bin_labels = batch["binary_label"].to(self.device)
            cat_labels = batch["category_label"].to(self.device)

            bin_logits, cat_logits = self.model(input_ids, attention_mask)

            bin_loss = self.binary_criterion(bin_logits, bin_labels)

            # Masked category loss (ignore_index=-1 handles the masking automatically)
            cat_loss = self.category_criterion(cat_logits, cat_labels)

            loss = bin_loss + self.lambda_cat * cat_loss
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_bin_loss += bin_loss.item()
            total_cat_loss += cat_loss.item()

        n = len(self.train_loader)
        return total_loss/n, total_bin_loss/n, total_cat_loss/n

    def evaluate(self, loader):
        self.model.eval()
        all_bin_preds, all_bin_labels = [], []
        all_cat_preds, all_cat_labels = [], []

        total_loss, total_bin_loss, total_cat_loss = 0, 0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bin_labels = batch["binary_label"].to(self.device)
                cat_labels = batch["category_label"].to(self.device)

                bin_logits, cat_logits = self.model(input_ids, attention_mask)

                bin_loss = self.binary_criterion(bin_logits, bin_labels)
                cat_loss = self.category_criterion(cat_logits, cat_labels)
                loss = bin_loss + self.lambda_cat * cat_loss

                total_loss += loss.item()
                total_bin_loss += bin_loss.item()
                total_cat_loss += cat_loss.item()

                bin_preds = torch.argmax(bin_logits, dim=1)
                cat_preds = torch.argmax(cat_logits, dim=1)

                all_bin_preds.extend(bin_preds.cpu().numpy())
                all_bin_labels.extend(bin_labels.cpu().numpy())

                all_cat_preds.extend(cat_preds.cpu().numpy())
                all_cat_labels.extend(cat_labels.cpu().numpy())

        # Binary metrics
        bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(all_bin_labels, all_bin_preds, average="macro")

        # Oracle category metrics (only where cat_labels != -1)
        valid_cat_idx = np.array(all_cat_labels) != -1
        valid_cat_preds = np.array(all_cat_preds)[valid_cat_idx]
        valid_cat_labels = np.array(all_cat_labels)[valid_cat_idx]

        if len(valid_cat_labels) > 0:
            cat_p, cat_r, cat_f1, _ = precision_recall_fscore_support(valid_cat_labels, valid_cat_preds, average="macro", zero_division=0)
        else:
            cat_p, cat_r, cat_f1 = 0, 0, 0

        metrics = {
            "loss": total_loss / len(loader),
            "bin_loss": total_bin_loss / len(loader),
            "cat_loss": total_cat_loss / len(loader),
            "bin_f1": bin_f1,
            "oracle_cat_f1": cat_f1,
            "combined_score": 0.5 * bin_f1 + 0.5 * cat_f1,
            "bin_preds": all_bin_preds,
            "bin_labels": all_bin_labels,
            "cat_preds": all_cat_preds,
            "cat_labels": all_cat_labels
        }
        return metrics

    def save_checkpoint(self, path, epoch, best_score, config, seed=42, patience_counter=0):
        import os
        import tempfile

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") and self.scheduler else None,
            "epoch": epoch,
            "best_score": best_score,
            "patience_counter": patience_counter,
            "config": config,
            "seed": seed
        }

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="tmp_chkpt_", suffix=".pt")
        with os.fdopen(fd, 'wb') as f:
            torch.save(checkpoint, f)

        os.replace(temp_path, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle legacy checkpoint (state_dict only)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            self.model.load_state_dict(checkpoint)
            return -1, -1.0, 0, {}, 42

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if hasattr(self, "scheduler") and self.scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", -1)
        best_score = checkpoint.get("best_score", -1.0)
        patience_counter = checkpoint.get("patience_counter", 0)
        config = checkpoint.get("config", {})
        seed = checkpoint.get("seed", 42)

        return epoch, best_score, patience_counter, config, seed
