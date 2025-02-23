import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from feature_extractor import FeatureExtractor


class MultiScaleLocalNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Conv1d(config.feature_dim, 256, kernel_size=k, padding=k // 2, bias=True, dtype=torch.bfloat16)
            for k in [3, 5, 7]
        ])
        self.pool = nn.ModuleList([
            nn.MaxPool1d(kernel_size=2 ** i, stride=2 ** i) for i in range(config.scale_num)
        ])
        self.activation = nn.GELU()

    def forward(self, features):
        features = features.transpose(1, 2)
        local_feats = []
        for conv, pool in zip(self.scales, self.pool):
            feat = self.activation(conv(features))
            feat = pool(feat)
            local_feats.append(feat)
        return local_feats


class EnhancedFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=config.num_heads,
            batch_first=True,
            dtype=torch.bfloat16
        )
        self.local_projection = nn.Linear(256, config.feature_dim, dtype=torch.bfloat16)
        self.project = nn.Linear(config.feature_dim, config.fusion_dim, dtype=torch.bfloat16)

    def forward(self, global_feat, local_feats, boundary_feat):
        print(f"global_feat shape: {global_feat.shape}")
        local_mean = [lf.mean(dim=2) for lf in local_feats]
        local_proj = [self.local_projection(lm) for lm in local_mean]
        local_fused = torch.stack(local_proj, dim=1).mean(dim=1)
        print(f"local_fused shape: {local_fused.shape}, boundary_feat shape: {boundary_feat.shape}")
        features = torch.stack([global_feat, local_fused, boundary_feat], dim=1)
        attn_output, _ = self.attention(features, features, features)
        fused = attn_output.mean(dim=1)
        return self.project(fused)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tcn = nn.Conv1d(config.feature_dim, config.feature_dim, kernel_size=3, padding=1, dtype=torch.bfloat16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.num_heads,
            dim_feedforward=config.fusion_dim,
            activation=F.gelu,
            norm_first=True,
            batch_first=True,
            dropout=0.1,
            dtype=torch.bfloat16
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.residual = nn.Linear(config.feature_dim, config.feature_dim, dtype=torch.bfloat16)

    def forward(self, x):
        x_tcn = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x_out = self.transformer(x + x_tcn)
        return x_out + self.residual(x)


class DetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(finetune=config.finetune_ssl)
        self.local_net = MultiScaleLocalNet(config)
        self.boundary_net = nn.Sequential(
            nn.Linear(config.feature_dim, 128, dtype=torch.bfloat16),
            nn.LayerNorm(128, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(128, 299, dtype=torch.bfloat16)  # Match segment_label size
        )
        self.transformer = TransformerEncoder(config)
        self.fusion = EnhancedFusion(config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.fusion_dim, dtype=torch.bfloat16),
            nn.Linear(config.fusion_dim, 1, dtype=torch.bfloat16)
        )

    def forward(self, waveforms):
        features = self.feature_extractor(waveforms)  # [batch_size, seq_len, feature_dim]
        local_features = self.local_net(features)  # List of [batch_size, 256, seq_len']

        # 对 features 在 seq_len 维度求平均，输入 boundary_net
        features_mean = features.mean(dim=1)  # [batch_size, feature_dim]
        boundary = self.boundary_net(features_mean)  # [batch_size, 299]
        boundary_feat = boundary.mean(dim=1)  # [batch_size]
        print(
            f"features shape: {features.shape}, boundary shape: {boundary.shape}, boundary_feat shape: {boundary_feat.shape}")
        boundary_feat = boundary_feat.unsqueeze(1).expand(-1, Config.feature_dim)  # [batch_size, feature_dim]

        global_feat = self.transformer(features).mean(dim=1)  # [batch_size, feature_dim]
        fused = self.fusion(global_feat, local_features, boundary_feat)
        logits = self.classifier(fused)
        return {
            'logits': logits,  # [batch_size, 1]
            'local': local_features,  # List of [batch_size, 256, seq_len']
            'boundary': boundary  # [batch_size, 299]
        }