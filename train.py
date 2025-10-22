import torch
import toml
import utils
import os

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from models import AnomalyDetectionModel
from dataset import AnomalyDetectionDataset


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pad_sequences(batch):
    batch_lengths = torch.tensor([item[2] for item in batch])
    batch_inputs = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]

    batch_inputs = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
    batch_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

    return batch_inputs, batch_labels, batch_lengths


def dice_weight(probability, delta=0.1):
    return (probability / delta).clamp(min=0.0, max=1.0)


def bidirectional_dice_loss(outputs, targets):
    return 1 - utils.bidirectional_dice_score(outputs, targets, dice_weight(targets.sum() / targets.numel()))


def criterion(outputs, targets, lengths):
    batch_loss = torch.tensor(0).to(lengths.device).float()

    for batch, length in enumerate(lengths):
        output = outputs[batch, :length]
        target = targets[batch, :length]

        batch_loss += nn.functional.binary_cross_entropy(output, target) + bidirectional_dice_loss(output, target)

    return batch_loss / lengths.shape[0]


# 加载配置
configs = toml.load('configs/config.toml')

# 设置随机种子
set_random_seed(configs['seed'])

# 创建检查点目录
os.makedirs('checkpoints', exist_ok=True)

# 加载数据集
full_dataset = AnomalyDetectionDataset(configs['train-data-path'])
dataset_size = len(full_dataset)

# 数据集划分
train_size = int(configs.get('train-split', 0.8) * dataset_size)
valid_size = dataset_size - train_size

train_dataset, valid_dataset = random_split(
    full_dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(configs['seed'])
)

print(f"\n==================== 数据集信息 ====================")
print(f"数据集路径: {configs['train-data-path']}")
print(f"总数据量: {dataset_size}")
print(f"训练集: {train_size} ({configs.get('train-split', 0.8)*100:.0f}%)")
print(f"验证集: {valid_size} ({(1-configs.get('train-split', 0.8))*100:.0f}%)")

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=configs['batch-size'],
    num_workers=configs['num-workers'],
    shuffle=True,
    collate_fn=pad_sequences
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=configs['group-size'],
    num_workers=configs['num-workers'],
    shuffle=False,
    collate_fn=pad_sequences
)

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

num_epochs = configs['num-epochs']

best_iou_score = 0.0
last_iou_score = 0.0

device = torch.device(configs['device'])

attention_window = configs['attention-window']
smoothing_window = configs['smoothing-window']
log_interval = configs['log-interval']

print(f"\n==================== 模型配置 ====================")
print(f"设备: {device}")
print(f"注意力窗口: {attention_window}")
print(f"Alpha参数: {configs['alpha']}")
print(f"平滑窗口: {smoothing_window}")

# 创建模型
model = AnomalyDetectionModel(attention_window, alpha=configs['alpha'])
model = model.to(device)

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=configs['learning-rate'],
    weight_decay=configs['weight-decay']
)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)

# 检查点路径
load_checkpoint_path = configs['load-checkpoint-path']
best_checkpoint_path = configs['best-checkpoint-path']
last_checkpoint_path = configs['last-checkpoint-path']

print(f"\n==================== 训练配置 ====================")
print(f"训练轮数: {num_epochs}")
print(f"批次大小: {configs['batch-size']}")
print(f"学习率: {configs['learning-rate']}")
print(f"权重衰减: {configs['weight-decay']}")
print(f"最佳模型保存路径: {best_checkpoint_path}")
print(f"最新模型保存路径: {last_checkpoint_path}")

# 加载检查点
if configs['load-checkpoint']:
    if os.path.exists(load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"✓ 已加载检查点: {load_checkpoint_path}")
    else:
        print(f"⚠ 检查点文件不存在: {load_checkpoint_path}")

print(f'\n==================== 开始训练 ====================\n')

for epoch in range(num_epochs):
    # ========== 训练阶段 ==========
    model.train()
    train_loss_sum = 0.0

    for batch, (inputs, labels, lengths) in enumerate(train_dataloader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.sigmoid(), labels, lengths)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        if batch % log_interval == 0:
            print(f'{utils.current_time()} [train] [{epoch+1:03d}/{num_epochs:03d}] [{batch:04d}/{train_dataloader_size:04d}] loss: {loss.item():.5f}')

    avg_train_loss = train_loss_sum / train_dataloader_size

    # ========== 验证阶段 ==========
    model.eval()

    with torch.no_grad():
        all_scores = []
        all_labels = []

        for index, (inputs, labels, lengths) in enumerate(valid_dataloader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            scores = model(inputs).sigmoid()
            scores = utils.score_smoothing(scores, smoothing_window)

            # 按实际长度截取有效部分
            for i, length in enumerate(lengths):
                all_scores.append(scores[i, :length])
                all_labels.append(labels[i, :length])

            if index % log_interval == 0:
                print(f'{utils.current_time()} [valid] [{epoch+1:03d}/{num_epochs:03d}] [{index:04d}/{valid_dataloader_size:04d}]')

        # 合并所有结果
        all_scores = torch.cat(all_scores).cpu()
        all_labels = torch.cat(all_labels).cpu()

        # 计算IoU
        iou_score = utils.iou_score((all_scores > 0.5).int(), all_labels).item()

        # 保存最佳模型
        if iou_score > best_iou_score:
            best_iou_score = iou_score
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f'{utils.current_time()} ✓ 保存最佳模型 IoU: {iou_score:.4f} -> {best_checkpoint_path}')

        # 保存最新模型
        last_iou_score = iou_score
        torch.save(model.state_dict(), last_checkpoint_path)

        # 更新学习率
        scheduler.step(iou_score)

    print(f'{utils.current_time()} [epoch {epoch+1:03d}/{num_epochs:03d}] loss: {avg_train_loss:.5f} | IoU: {iou_score:.4f} | best: {best_iou_score:.4f}\n')

print(f'\n==================== 训练完成 ====================')
print(f'最佳 IoU: {best_iou_score:.4f}')
print(f'最终 IoU: {last_iou_score:.4f}')
print(f'最佳模型: {best_checkpoint_path}')
print(f'最新模型: {last_checkpoint_path}\n')