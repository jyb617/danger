import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance as distance


def create_position_mask(attention_window, sequence_length):
    """动态创建位置掩码"""
    position_mask = torch.ones(sequence_length, sequence_length)

    for index in range(sequence_length):
        lower_bound = max(index - attention_window // 2, 0)
        upper_bound = min(index + attention_window // 2 + 1, sequence_length)
        position_mask[index, lower_bound:upper_bound] = 0

    return position_mask.bool()


def create_position_info(sequence_length):
    """动态创建位置信息"""
    positions = np.arange(sequence_length).reshape(-1, 1)
    positions = distance.pdist(positions)
    positions = distance.squareform(positions)
    return torch.from_numpy(positions ** 2).float()


class ContextAttention(nn.Module):
    def __init__(self, in_features, attention_window, alpha=0.5, embedding_features=128, num_heads=1):
        super().__init__()
        # ✅ 移除固定的位置编码，改为动态生成
        self.attention_window = attention_window

        self.project_k = nn.Linear(in_features=in_features, out_features=embedding_features)
        self.project_q = nn.Linear(in_features=in_features, out_features=embedding_features)
        self.project_v = nn.Linear(in_features=in_features, out_features=embedding_features)

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        self.output_project = nn.Linear(in_features=embedding_features, out_features=in_features)

        self.num_heads = num_heads
        self.alpha = alpha
        self.scale = embedding_features ** 0.5

        # ✅ 缓存位置编码，避免重复计算
        self.cached_position_mask = {}
        self.cached_position_info = {}

    def get_position_encodings(self, sequence_length, device):
        """获取或创建位置编码（带缓存）"""
        # 检查缓存
        if sequence_length not in self.cached_position_mask:
            self.cached_position_mask[sequence_length] = create_position_mask(
                self.attention_window, sequence_length
            )
            self.cached_position_info[sequence_length] = create_position_info(
                sequence_length
            )

        position_mask = self.cached_position_mask[sequence_length].to(device)
        position_info = self.cached_position_info[sequence_length].to(device)

        return position_mask, position_info

    def forward(self, inputs):
        sequence_length = inputs.shape[1]

        # ✅ 动态获取位置编码
        position_mask, position_info = self.get_position_encodings(
            sequence_length, inputs.device
        )

        q = self.project_q(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        positional_encoding = torch.exp(-(self.gamma * position_info - self.theta).abs())

        attention_map1 = (q @ k) / self.scale + positional_encoding
        attention_map2 = attention_map1.masked_fill(position_mask, -1e9)

        attention_map1 = attention_map1.softmax(dim=3)
        attention_map2 = attention_map2.softmax(dim=3)

        output1 = attention_map1 @ v
        output2 = attention_map2 @ v
        outputs = self.alpha * output1 + (1 - self.alpha) * output2

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous()
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)

        return self.output_project(outputs)


class AnomalyDetectionModel(nn.Module):
    def __init__(self, attention_window=5, alpha=0.5):
        super().__init__()
        self.attention = ContextAttention(
            in_features=128,
            attention_window=attention_window,
            alpha=alpha
        )

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)

        self.fc1 = nn.Linear(in_features=2304, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.1)

        self.classifier = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, padding=0)
        self.gelu = nn.GELU()
        self.padding = nn.ZeroPad1d((2, 0))

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.gelu(outputs)
        outputs = self.norm1(outputs)
        outputs = self.dropout1(outputs)

        outputs = self.fc2(outputs)
        outputs = self.gelu(outputs)
        outputs = self.norm2(outputs)
        outputs = self.dropout2(outputs)

        outputs = outputs + self.attention(outputs)
        outputs = self.norm3(outputs)

        outputs = self.padding(outputs.transpose(1, 2))
        outputs = self.classifier(outputs)

        return outputs.flatten(1)