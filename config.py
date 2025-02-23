import os


class Config:
    # 数据配置
    sample_rate = 16000
    max_duration = 6.0  # 秒
    max_audio_length = int(sample_rate * max_duration)  # 96000 samples
    min_audio_length = int(sample_rate * 2.0)  # 32000 samples

    # 模型参数
    feature_dim = 768  # wav2vec2-base 输出维度
    transformer_layers = 4  # 减少层数以减轻过拟合
    num_heads = 8
    temporal_embed_dim = 256
    fusion_dim = 512  # 调整融合维度
    scale_num = 3
    ssl_model = "wav2vec2-base-960h"
    finetune_ssl = True  # 启用微调

    # 训练参数
    batch_size = 16
    lr = 1e-4  # 降低初始学习率
    weight_decay = 1e-2
    epochs = 10  # 增加训练轮数
    warmup_ratio = 0.2  # 延长预热期

    # 路径配置
    audio_dir = None
    list_path = None
    label_path = None
    dataset_type = None

    @staticmethod
    def set_paths(audio_dir, list_path, label_path, dataset_type):
        """设置路径参数并规范化"""
        Config.audio_dir = os.path.normpath(audio_dir)
        Config.list_path = os.path.normpath(list_path)
        Config.label_path = os.path.normpath(label_path)
        Config.dataset_type = dataset_type