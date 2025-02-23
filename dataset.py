import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from config import Config


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
            self.labels = {os.path.splitext(f)[0]: np.array([0], dtype=np.int32) for f in self.file_list}

        missing = set(self.file_list) - set(self.labels.keys())
        if missing:
            print(f"Warning: {len(missing)} files missing labels")
            for f in missing:
                self.labels[os.path.splitext(f)[0]] = np.array([0], dtype=np.int32)

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
            print(f"Error loading audio file {file_path}: {e}")
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
