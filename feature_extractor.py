import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

model_path = "E:/DataSets/facebook/wav2vec2-base-960h/"


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=model_path, finetune=True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_path, local_files_only=True).to(torch.bfloat16)
        self.weight_hidd = nn.Parameter(torch.zeros(12, dtype=torch.bfloat16))
        if finetune:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, waveforms, attention_mask=None):
        inputs = {
            'input_values': waveforms.to(self.model.device, dtype=torch.bfloat16),
            'attention_mask': attention_mask.to(self.model.device, dtype=torch.bfloat16) if attention_mask else None
        }
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        stacked = torch.stack(hidden_states, dim=0)
        norm_weights = F.softmax(self.weight_hidd, dim=-1)
        weighted = (norm_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=0)
        return weighted.to(torch.bfloat16)
