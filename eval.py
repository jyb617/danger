import torch
import toml
import utils
import numpy as np

from torch.utils.data import DataLoader

from models import AnomalyDetectionModel
from dataset import AnomalyDetectionDataset


def f1_score_manual(scores, labels, threshold):
    """手动实现F1 score，避免sklearn依赖"""
    pred = (scores > threshold).int()

    tp = (pred * labels).sum().float()
    fp = (pred * (1 - labels)).sum().float()
    fn = ((1 - pred) * labels).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.item()


def auc_score_manual(labels, scores):
    """手动实现简单的AUC计算"""
    # 将数据转为numpy
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    # 简单的AUC近似计算
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []

    for threshold in thresholds:
        pred = (scores > threshold).astype(int)

        tp = np.sum(pred * labels)
        fp = np.sum(pred * (1 - labels))
        tn = np.sum((1 - pred) * (1 - labels))
        fn = np.sum((1 - pred) * labels)

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        tprs.append(tpr)
        fprs.append(fpr)

    # 计算AUC（梯形法则）
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # 排序
    sorted_indices = np.argsort(fprs)
    fprs = fprs[sorted_indices]
    tprs = tprs[sorted_indices]

    auc = np.trapezoid(tprs, fprs)
    return auc


def average_precision_manual(labels, scores):
    """手动实现AP计算"""
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    # 计算precision和recall
    tp = np.cumsum(sorted_labels)
    total_positives = np.sum(labels)

    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / total_positives

    # 计算AP
    ap = 0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]

    return ap


def far_score(scores, labels, threshold):
    return utils.far_score(scores, labels, threshold)


def iou_score(scores, labels, threshold):
    return utils.iou_score((scores > threshold).int(), labels)


configs = toml.load('configs/config.toml')

# 修改数据集路径以适应新的数据格式
dataset = AnomalyDetectionDataset('D:/Python/dataprocess/processed_data')
dataset_size = len(dataset)

if dataset_size == 0:
    print("错误：数据集为空，请检查数据路径")
    exit(1)

# 创建数据加载器，适应小数据集
batch_size = min(configs.get('group-size', 1), dataset_size)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

attention_window = configs['attention-window']
smoothing_window = configs['smoothing-window']

log_interval = configs['log-interval']

model = AnomalyDetectionModel(attention_window, alpha=configs['alpha'])
model = model.to(device)

print(f'\n---------- evaluation start at: {device} ----------\n')
print(f'数据集大小: {dataset_size}')
print(f'批次数量: {dataloader_size}')


# 添加填充函数
def pad_sequences(batch):
    batch_lengths = torch.tensor([item[2] for item in batch])
    batch_inputs = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
    return batch_inputs, batch_labels, batch_lengths


dataloader.collate_fn = pad_sequences

with torch.no_grad():
    scores0 = []
    scores1 = []

    labels0 = []
    labels1 = []

    model.load_state_dict(torch.load(configs['load-checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for index, (inputs, labels, lengths) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = model(inputs).sigmoid()
        scores = utils.score_smoothing(scores, smoothing_window)

        # 修改：移除repeat_interleave，适应新的数据格式
        # scores = utils.score_smoothing(scores, smoothing_window).repeat_interleave(16, dim=1)

        # 按实际序列长度处理每个样本
        for i, length in enumerate(lengths):
            sample_scores = scores[i, :length]
            sample_labels = labels[i, :length]

            if sample_labels.sum() == 0:
                scores0.append(sample_scores)
                labels0.append(sample_labels)
            else:
                scores1.append(sample_scores)
                labels1.append(sample_labels)

        if index % log_interval == 0:
            print(f'{utils.current_time()} [eval] [{index:04d}/{dataloader_size:04d}]')

    # 检查是否有数据
    if len(scores0) == 0 and len(scores1) == 0:
        print("错误：没有找到任何有效数据")
        exit(1)

    # 合并数据
    if len(scores0) > 0:
        scores0 = torch.cat(scores0).cpu()
        labels0 = torch.cat(labels0).cpu()
    else:
        scores0 = torch.tensor([])
        labels0 = torch.tensor([])

    if len(scores1) > 0:
        scores1 = torch.cat(scores1).cpu()
        labels1 = torch.cat(labels1).cpu()
    else:
        scores1 = torch.tensor([])
        labels1 = torch.tensor([])

    scores = torch.cat([scores0, scores1])
    labels = torch.cat([labels0, labels1])

    print(f'\n数据统计:')
    print(f'正常样本数量: {len(scores0)}')
    print(f'异常样本数量: {len(scores1)}')
    print(f'总样本数量: {len(scores)}')

    # 使用手动实现的指标
    auc_score = auc_score_manual(labels, scores)

    f1_score20 = f1_score_manual(scores, labels, 0.2)
    f1_score30 = f1_score_manual(scores, labels, 0.3)
    f1_score40 = f1_score_manual(scores, labels, 0.4)
    f1_score50 = f1_score_manual(scores, labels, 0.5)

    ap_score = average_precision_manual(labels, scores)

    # FAR和IoU计算（需要分别在正常和异常样本上计算）
    if len(scores0) > 0:
        far20 = far_score(scores0, labels0, 0.2)
        far30 = far_score(scores0, labels0, 0.3)
        far40 = far_score(scores0, labels0, 0.4)
        far50 = far_score(scores0, labels0, 0.5)
    else:
        far20 = far30 = far40 = far50 = 0.0

    if len(scores1) > 0:
        iou20 = iou_score(scores1, labels1, 0.2)
        iou30 = iou_score(scores1, labels1, 0.3)
        iou40 = iou_score(scores1, labels1, 0.4)
        iou50 = iou_score(scores1, labels1, 0.5)
    else:
        iou20 = iou30 = iou40 = iou50 = 0.0

    print('\n--------------------------------')
    print(f'F1-Score@20: {f1_score20:.4f}')
    print(f'F1-Score@30: {f1_score30:.4f}')
    print(f'F1-Score@40: {f1_score40:.4f}')
    print(f'F1-Score@50: {f1_score50:.4f}')

    print('\n--------------------------------')
    print(f'FAR@20: {far20:.4f}')
    print(f'FAR@30: {far30:.4f}')
    print(f'FAR@40: {far40:.4f}')
    print(f'FAR@50: {far50:.4f}')

    print('\n--------------------------------')
    print(f'IoU@20: {iou20:.4f}')
    print(f'IoU@30: {iou30:.4f}')
    print(f'IoU@40: {iou40:.4f}')
    print(f'IoU@50: {iou50:.4f}')

    print(f'\nAUC: {auc_score:<8.4f} AP: {ap_score:.4f}')

print(f'\n---------- evaluation finished ----------\n')