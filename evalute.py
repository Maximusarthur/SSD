import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import Config
from model import DetectionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchaudio
import numpy as np
import logging

logging.basicConfig(filename='evalute.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


class PartialSpoofDataset(Dataset):
    def __init__(self, audio_dir, list_path, label_npy_path, transform=None):
        self.audio_dir = os.path.normpath(audio_dir)
        self.transform = transform

        with open(list_path, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        try:
            labels = np.load(label_npy_path, allow_pickle=True).item()
            self.labels = {k: np.array(v, dtype=np.int32) for k, v in labels.items()}
        except Exception as e:
            print(f"Error loading labels: {e}")
            logging.error(f"Error loading labels: {e}")
            self.labels = {os.path.splitext(f)[0]: np.array([0], dtype=np.int32) for f in self.file_list}

        missing = set(self.file_list) - set(self.labels.keys())
        if missing:
            print(f"Warning: {len(missing)} files missing labels")
            logging.warning(f"{len(missing)} files missing labels")
            for f in missing:
                self.labels[os.path.splitext(f)[0]] = np.array([0], dtype=np.int32)

        label_values = [self.labels[os.path.splitext(f)[0]][0] for f in self.file_list]
        label_info = f"Label distribution: Positive={sum(label_values)} / Total={len(label_values)}"
        print(label_info)
        logging.info(label_info)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        if not file_name.lower().endswith(".wav"):
            file_name += ".wav"
        file_path = os.path.join(self.audio_dir, file_name)

        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            error_msg = f"Error loading audio file {file_path}: {e}"
            print(error_msg)
            logging.error(error_msg)
            waveform = torch.zeros(1, Config.max_audio_length, dtype=torch.float32)
            sample_rate = Config.sample_rate

        if sample_rate != Config.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=Config.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        current_length = waveform.size(1)
        if current_length > Config.max_audio_length:
            start = torch.randint(0, current_length - Config.max_audio_length + 1, (1,))
            waveform = waveform[:, start:start + Config.max_audio_length]
        else:
            pad_amount = Config.max_audio_length - current_length
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0.0)

        key = os.path.splitext(os.path.basename(file_name))[0]
        label = self.labels.get(key, np.array([0], dtype=np.int32))
        classification_label = torch.tensor(label[0], dtype=torch.float32)
        seq_len = 299
        segment_label = self._adjust_segment_label(label, seq_len)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0), classification_label, segment_label

    def _adjust_segment_label(self, label, target_len, frame_shift=160):
        if len(label) <= 1:
            return torch.zeros(target_len, dtype=torch.float32)
        segment_label = torch.from_numpy(label[1:]).float()
        if segment_label.size(0) != target_len:
            num_frames = int(segment_label.size(0) / frame_shift)
            if num_frames == 0:
                return torch.zeros(target_len)
            segment_label = F.interpolate(
                segment_label.unsqueeze(0).unsqueeze(0),
                size=target_len,
                mode='nearest'
            ).squeeze(0).squeeze(0)
        return segment_label


def collate_fn(batch):
    waveforms, classification_labels, segment_labels = zip(*batch)
    waveforms = torch.stack(waveforms)
    classification_labels = torch.stack(classification_labels)
    segment_labels = torch.stack(segment_labels)
    return waveforms, classification_labels, segment_labels


def evaluate(model, loader, device, thresholds=[0.5]):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            waveforms, labels, _ = batch
            waveforms = waveforms.to(device, dtype=torch.bfloat16)
            labels = labels.to(device, dtype=torch.bfloat16).unsqueeze(1)

            outputs = model(waveforms)
            logits = outputs['logits']

            all_preds.extend((torch.sigmoid(logits) > thresholds[0]).float().cpu().float().numpy())
            all_labels.extend(labels.cpu().float().numpy())
            all_logits.extend(logits.cpu().float().numpy())

    debug_info = [
        f"Predicted positives (default threshold {thresholds[0]}): {sum(all_preds)} / {len(all_preds)}",
        f"True positives: {sum(all_labels)} / {len(all_labels)}",
        f"Logits range: min={min(all_logits)}, max={max(all_logits)}"
    ]
    for info in debug_info:
        print(info)
        logging.info(info)

    results = {}
    for threshold in thresholds:
        preds = (torch.sigmoid(torch.tensor(all_logits)) > threshold).float().numpy()
        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        results[threshold] = (accuracy, precision, recall, f1)
        print(f"\nThreshold: {threshold}")
        print(f"Predicted positives: {sum(preds)} / {len(preds)}")
        logging.info(f"Threshold: {threshold}, Predicted positives: {sum(preds)} / {len(preds)}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return results


def main():
    config = Config()
    config.set_paths(
        audio_dir="E:/DataSets/PartialSpoof_v1.2/database/eval/con_wav",  # 修正为 eval 集
        list_path="E:/DataSets/PartialSpoof_v1.2/database/eval/eval.lst",
        label_path="E:/DataSets/PartialSpoof_v1.2/database/segment_labels/eval_seglab_0.64.npy"
    )

    try:
        dataset = PartialSpoofDataset(config.audio_dir, config.list_path, config.label_path)
    except Exception as e:
        error_msg = f"Error initializing dataset: {e}"
        print(error_msg)
        logging.error(error_msg)
        return

    loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                        num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True)

    model = DetectionModel(config).to(device)
    try:
        model.load_state_dict(torch.load("models/model_final.pth"))
        print("Model loaded successfully.")
        logging.info("Model loaded successfully.")
        weights = model.classifier[1].weight.data[:5]
        print("Sample classifier weights:", weights)
        logging.info(f"Sample classifier weights: {weights}")
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        print(error_msg)
        logging.error(error_msg)
        return

    try:
        # 测试更适合当前 logits 范围的阈值
        results = evaluate(model, loader, device, thresholds=[0.5, 0.45, 0.4, 0.35, 0.3])
    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        print(error_msg)
        logging.error(error_msg)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Type:", device)
    logging.info(f"Device Type: {device}")
    main()
