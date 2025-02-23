import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from config import Config
from dataset import PartialSpoofDataset, collate_fn
from model import DetectionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, config):
        self.config = config
        self.class_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.5], device=device))
        self.segment_criterion = nn.BCEWithLogitsLoss()
        self.local_criterion = nn.MSELoss()
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))

    def focal_loss(self, logits, labels, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-bce)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.mean()

    def train_epoch(self, model, loader, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            waveforms, labels, segment_labels = batch
            waveforms = waveforms.to(device, dtype=torch.bfloat16)
            labels = labels.to(device, dtype=torch.bfloat16).unsqueeze(1)
            segment_labels = segment_labels.to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()
            outputs = model(waveforms)

            class_loss = self.focal_loss(outputs['logits'], labels)
            seg_loss = self.segment_criterion(outputs['boundary'], segment_labels)
            local_loss = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
            for lf in outputs['local']:
                lf_mean = lf.mean(dim=2)
                boundary_mean = outputs['boundary'].mean(dim=1).unsqueeze(-1).expand(-1, 256)
                local_loss += self.local_criterion(lf_mean, boundary_mean)
            local_loss = local_loss / len(outputs['local'])

            loss = (class_loss / (2 * torch.exp(self.log_vars[0])) + self.log_vars[0].mean() +
                    seg_loss / (2 * torch.exp(self.log_vars[1])) + self.log_vars[1].mean() +
                    local_loss / (2 * torch.exp(self.log_vars[2])) + self.log_vars[2].mean())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), class_loss=class_loss.item(), seg_loss=seg_loss.item())
        scheduler.step()
        return total_loss / len(loader)


def evaluate(model, loader, device, thresholds=[0.5]):
    model.eval()
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            waveforms, labels, _ = batch
            waveforms = waveforms.to(device, dtype=torch.bfloat16)
            labels = labels.to(device, dtype=torch.bfloat16).unsqueeze(1)
            outputs = model(waveforms)
            logits = outputs['logits']
            all_labels.extend(labels.cpu().float().numpy())
            all_logits.extend(logits.cpu().float().numpy())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    results = {}
    for threshold in thresholds:
        preds = (probs > threshold).astype(float)
        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        results[threshold] = (accuracy, precision, recall, f1)
        print(
            f"Threshold {threshold}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return results


def main():
    config = Config()
    config.set_paths(
        audio_dir="E:/DataSets/PartialSpoof_v1.2/database/train/con_wav",
        list_path="E:/DataSets/PartialSpoof_v1.2/database/train/train.lst",
        label_path="E:/DataSets/PartialSpoof_v1.2/database/segment_labels/train_seglab_0.64.npy",
        dataset_type="train"
    )
    train_dataset = PartialSpoofDataset(config.audio_dir, config.list_path, config.label_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                              num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True)

    config.set_paths(
        audio_dir="E:/DataSets/PartialSpoof_v1.2/database/dev/con_wav",
        list_path="E:/DataSets/PartialSpoof_v1.2/database/dev/dev.lst",
        label_path="E:/DataSets/PartialSpoof_v1.2/database/segment_labels/dev_seglab_0.64.npy",
        dataset_type="dev"
    )
    dev_dataset = PartialSpoofDataset(config.audio_dir, config.list_path, config.label_path)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                            num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True)

    model = DetectionModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup_steps = int(len(train_loader) * config.warmup_ratio)
    scheduler_warmup = LambdaLR(optimizer, lambda step: min(step / warmup_steps, 1.0))
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - warmup_steps // len(train_loader))
    scheduler_reduce = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    trainer = Trainer(config)
    torch.autograd.set_detect_anomaly(False)

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0
    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(model, train_loader, optimizer, scheduler_warmup, epoch + 1)
        if epoch >= warmup_steps // len(train_loader):
            scheduler_cosine.step()
        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}")

        val_results = evaluate(model, dev_loader, device, thresholds=[0.5])
        val_f1 = val_results[0.5][3]
        print(f"Validation F1: {val_f1:.4f}")
        scheduler_reduce.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path}")

    final_save_path = os.path.join(save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved at {final_save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Type:", device)
    print("Starting training...")
    main()
