
import argparse
import os
import time
from typing import List, Dict

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from dataset.pyg import MultiModalTransactionDataset
from pretrain import Model
from settings import HETERO_GRAPH_METADATA
from utils.transform import TextFeatEmbedding, format_data_type


def get_eval_report(
        x_train: np.ndarray, x_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        targets: List,
):
    det_model = RandomForestClassifier(
        n_estimators=300,
        n_jobs=os.cpu_count() // 2,
        class_weight={i: 1 if i == 0 else 20 for i in range(len(targets))},
    )
    det_model.fit(x_train, y_train)
    y_pred = det_model.predict(x_test)
    report = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=targets,
        output_dict=True
    )
    report['auc'] = float(roc_auc_score(
        y_true=[1 if item == 0 else -1 for item in y_test],
        y_score=[1 if item == 0 else -1 for item in y_pred],
    ))
    return report


def few_shot_eval(
        feats: np.ndarray,
        y_true: np.ndarray,
        targets: List[str],
        shots: int = 1,
) -> Dict:
    x_train, x_test, y_train, y_test = train_test_split(
        feats, y_true, test_size=0.2, shuffle=True,
    )
    x_train = np.concatenate([
        x_train[y_train == 0],
        *[x_train[y_train == i][:shots] for i in range(1, len(targets) + 1)],
    ], axis=0)
    y_train = np.concatenate([
        y_train[y_train == 0],
        *[y_train[y_train == i][:shots] for i in range(1, len(targets) + 1)],
    ], axis=0)

    report = get_eval_report(
        x_train=x_train, x_test=x_test,
        y_train=y_train, y_test=y_test,
        targets=targets,
    )
    return report


def infer(
        model: Model,
        text_emb: TextFeatEmbedding,
        device: torch.device,
        data: HeteroData
) -> List:
    with torch.no_grad():
        data = format_data_type(data)
        data = data.to(device)
        data = text_emb(data)
        graph_feats = model.graph_model(data)
        encoded_graph_feats = model.encoder(graph_feats)
    return encoded_graph_feats


def main(model_path: str, data_path: str, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Model(
        hidden_channels=kwargs.get('hidden_channels', 384),
        out_channels=kwargs.get('out_channels', 256),
        num_layers=kwargs.get('num_layers', 3),
        num_heads=kwargs.get('num_heads', 6),
        metadata=HETERO_GRAPH_METADATA,
        device=device,
    )
    model.load_state_dict(
        state_dict=torch.load(model_path, map_location=torch.device('cpu')),
        strict=False,
    )
    model = model.to(device)
    text_emb = TextFeatEmbedding().to(device)
    model.eval()
    text_emb.eval()

    dataset = MultiModalTransactionDataset(root=data_path, signature_path='/home/fm/www/misc/SignItem.csv') #传入函数签名

    y_true, feats = [], []
    targets = {
        cls: i for i, cls in enumerate([
            'Non-attack', 'reentrancy',
            'integer-overflow',
            'call-injection', 'honeypot',
            'flashloan attack',
        ])
    }
    nonattack_cnt = 100000
    batch_cache, batch_size = list(), kwargs.get('batch_size', 128)


    inference_time = 0.0
    
    for data in tqdm(dataset, total=len(dataset), desc='inferring'):
        if data.label not in targets:
            continue
        label_idx = targets[data.label]
        if label_idx == 0:
            if nonattack_cnt < 1:
                continue
            nonattack_cnt -= 1
        y_true.append(label_idx)
        batch_cache.append(data)

        if len(batch_cache) < batch_size:
            continue

        
        data = Batch.from_data_list(batch_cache)
        batch_cache = list()
        start_time = time.time()
        graph_embeds = infer(model, text_emb, device, data)
        inference_time += time.time() - start_time  # 累加推理时间
        print(inference_time)
        feats.extend(graph_embeds.detach().cpu().numpy())
    # 最后一个 batch
    if len(batch_cache) > 0:
        data = Batch.from_data_list(batch_cache)
        start_time = time.time()
        graph_embeds = infer(model, text_emb, device, data)
        inference_time += time.time() - start_time
        feats.extend(graph_embeds.detach().cpu().numpy())

    print(f"\n[推理阶段] Total inference time: {inference_time:.2f} seconds")


    targets = list(targets.keys())
    feats, y_true = np.array(feats), np.array(y_true)
    report_keys = targets + ['macro avg', 'weighted avg']

    start_time = time.time()
    for shots in [3, 5, 10, 30]:
        metrics_repeats = {
            key: {metric: list() for metric in [
                'precision', 'recall', 'f1-score', 'support'
            ]} for key in report_keys
        }
        metrics_repeats['auc'] = list()
        for _ in range(10):
            metrics = few_shot_eval(
                feats=feats,
                y_true=y_true,
                targets=targets,
                shots=shots,
            )
            for key in report_keys:
                for metric, val in metrics[key].items():
                    metrics_repeats[key][metric].append(val)
            metrics_repeats['auc'].append(metrics['auc'])
        print(f'{shots}-shot')
        print(metrics_repeats)

    metrics_repeats = {
        key: {metric: list() for metric in [
            'precision', 'recall', 'f1-score', 'support'
        ]} for key in report_keys
    }
    metrics_repeats['auc'] = list()
    for _ in range(10):
        x_train, x_test, y_train, y_test = train_test_split(
            feats, y_true, test_size=0.2, shuffle=True
        )
        metrics = get_eval_report(
            x_train=x_train, x_test=x_test,
            y_train=y_train, y_test=y_test,
            targets=targets,
        )
        for key in report_keys:
            for metric, val in metrics[key].items():
                metrics_repeats[key][metric].append(val)
        metrics_repeats['auc'].append(metrics['auc'])
    print('full-supervised')
    print(metrics_repeats)

    end_time = time.time()
    prediction_time = end_time - start_time


    print(f"\n[推理阶段] Total inference time: {inference_time:.2f} seconds")
    print(f"[预测阶段] Total prediction time: {prediction_time:.2f} seconds")
    print(f"[总体运行时间] Total: {inference_time + prediction_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=384)
    parser.add_argument('--out_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
    )
