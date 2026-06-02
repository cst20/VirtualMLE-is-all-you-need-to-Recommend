#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DOMAIN_CONFIGS = {
    'baby': {'folder_name': 'Baby', 'data_file': 'sequential_data_processed.txt'},
    'beauty': {'folder_name': 'Beauty', 'data_file': 'sequential_data_processed.txt'},
    'pet_supplies': {'folder_name': 'Pet_Supplies', 'data_file': 'sequential_data_processed.txt'},
    'movielens': {'folder_name': 'MovieLens', 'data_file': 'sequential_data_processed.txt'},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train/evaluate a minimal SASRec baseline')
    parser.add_argument('--domain', type=str, default='beauty', choices=sorted(DOMAIN_CONFIGS.keys()))
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--train_targets', type=str, default='all_positions', choices=['all_positions', 'last_position'])
    parser.add_argument('--train_window_size', type=int, default=0)
    parser.add_argument('--num_negatives', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260514)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--best_val_metric', type=str, default='Recall@10', choices=['Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10'])
    parser.add_argument('--output_json', type=str, default='')
    return parser.parse_args()


def resolve_domain_paths(script_dir: Path, domain: str, data_path_arg: str) -> Tuple[Path, Path]:
    cfg = DOMAIN_CONFIGS[str(domain).strip().lower()]
    domain_dir = script_dir / cfg['folder_name']
    default_data_path = domain_dir / cfg['data_file']
    data_path = Path(data_path_arg) if str(data_path_arg).strip() else default_data_path
    return domain_dir, data_path


def load_sequences(path: Path) -> Tuple[Dict[int, List[int]], int, int]:
    user_sequences: Dict[int, List[int]] = {}
    max_user = 0
    max_item = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            user = int(parts[0])
            seq = [int(x) for x in parts[1:]]
            if len(seq) < 3:
                continue
            user_sequences[user] = seq
            max_user = max(max_user, user)
            max_item = max(max_item, max(seq))
    return user_sequences, max_user, max_item


def count_training_stats(
    user_sequences: Dict[int, List[int]],
    max_len: int,
    train_window_size: int,
    train_targets: str,
) -> Dict[str, int]:
    total_interactions = 0
    train_interactions = 0
    training_input_tokens = 0
    supervised_targets = 0
    train_users = 0
    effective_window = max_len if train_window_size <= 0 else min(max_len, train_window_size)
    for seq in user_sequences.values():
        total_interactions += len(seq)
        if len(seq) < 3:
            continue
        train_seq = seq[:-2]
        if len(train_seq) < 2:
            continue
        train_users += 1
        train_interactions += len(train_seq)
        used_history = min(len(train_seq) - 1, effective_window)
        training_input_tokens += used_history
        if train_targets == 'all_positions':
            supervised_targets += used_history
        else:
            supervised_targets += 1
    return {
        'num_interactions_total': total_interactions,
        'train_users': train_users,
        'train_interactions': train_interactions,
        'training_input_tokens': training_input_tokens,
        'supervised_targets': supervised_targets,
    }


@dataclass
class UserSplit:
    train: List[int]
    val: int
    test: int
    full: set[int]


def build_leave_one_out_splits(user_sequences: Dict[int, List[int]]) -> Dict[int, UserSplit]:
    splits: Dict[int, UserSplit] = {}
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        splits[user] = UserSplit(train=seq[:-2], val=seq[-2], test=seq[-1], full=set(seq))
    return splits


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        splits: Dict[int, UserSplit],
        max_len: int,
        train_window_size: int,
        train_targets: str,
    ) -> None:
        self.users = [user for user, split in splits.items() if len(split.train) >= 2]
        self.splits = splits
        self.max_len = max_len
        self.train_window_size = train_window_size
        self.train_targets = train_targets

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        train = self.splits[user].train
        seq = np.zeros(self.max_len, dtype=np.int64)
        effective_window = self.max_len if self.train_window_size <= 0 else min(self.max_len, self.train_window_size)
        if self.train_targets == 'last_position':
            history = train[:-1][-effective_window:]
            target = train[-1]
            seq[self.max_len - len(history):] = np.array(history, dtype=np.int64)
            return torch.from_numpy(seq), torch.tensor(target, dtype=torch.long)
        pos = np.zeros(self.max_len, dtype=np.int64)
        history = train[-(effective_window + 1):]
        src = history[:-1]
        tgt = history[1:]
        seq[self.max_len - len(src):] = np.array(src, dtype=np.int64)
        pos[self.max_len - len(tgt):] = np.array(tgt, dtype=np.int64)
        return torch.from_numpy(seq), torch.from_numpy(pos)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x.transpose(-1, -2))
        out = F.gelu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        return out.transpose(-1, -2)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.attn_norm(x)
        attn_out, _ = self.attn(
            q,
            q,
            q,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.attn_dropout(attn_out)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        residual = x
        y = self.ffn_norm(x)
        y = self.ffn(y)
        x = residual + self.ffn_dropout(y)
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.output_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()
            self.output_embedding.weight[self.output_embedding.padding_idx].zero_()

    def output_weight(self) -> torch.Tensor:
        return self.output_embedding.weight[1:]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_embedding(seq) * math.sqrt(self.hidden_dim)
        x = self.emb_dropout(x + self.pos_embedding(positions))
        key_padding_mask = seq.eq(0)
        attn_mask = torch.triu(torch.ones((seq.size(1), seq.size(1)), device=seq.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

    def predict_scores(self, seq: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(seq)
        final_hidden = hidden[:, -1, :]
        return final_hidden @ self.output_weight().t()


def sampled_softmax_loss(
    hidden: torch.Tensor,
    target_index: torch.Tensor,
    output_weight: torch.Tensor,
    num_negatives: int,
) -> torch.Tensor:
    if num_negatives <= 0:
        logits = hidden @ output_weight.t()
        return F.cross_entropy(logits, target_index)

    num_classes = output_weight.size(0)
    neg_index = torch.randint(0, num_classes, (target_index.size(0), num_negatives), device=target_index.device)
    target_expanded = target_index.unsqueeze(1)
    collision_mask = neg_index.eq(target_expanded)
    while collision_mask.any():
        neg_index[collision_mask] = torch.randint(0, num_classes, (int(collision_mask.sum().item()),), device=target_index.device)
        collision_mask = neg_index.eq(target_expanded)

    pos_weight = output_weight[target_index]
    neg_weight = output_weight[neg_index]
    pos_logits = (hidden * pos_weight).sum(dim=-1, keepdim=True)
    neg_logits = torch.einsum('bd,bnd->bn', hidden, neg_weight)
    sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
    sampled_target = torch.zeros(target_index.size(0), dtype=torch.long, device=target_index.device)
    return F.cross_entropy(sampled_logits, sampled_target)


def build_eval_sequences(
    splits: Dict[int, UserSplit],
    users: Sequence[int],
    max_len: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    targets = []
    for user in users:
        split = splits[user]
        if mode == 'val':
            history = split.train
            target = split.val
        elif mode == 'test':
            history = split.train + [split.val]
            target = split.test
        else:
            raise ValueError(f'Unknown eval mode: {mode}')
        arr = np.zeros(max_len, dtype=np.int64)
        history = history[-max_len:]
        arr[max_len - len(history):] = np.array(history, dtype=np.int64)
        seqs.append(arr)
        targets.append(target)
    return torch.from_numpy(np.stack(seqs)), torch.tensor(targets, dtype=torch.long)


def evaluate_exact(
    model: SASRec,
    splits: Dict[int, UserSplit],
    users: Sequence[int],
    max_len: int,
    eval_batch_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, float]:
    model.eval()
    hit5 = hit10 = ndcg5 = ndcg10 = 0.0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(users), eval_batch_size):
            batch_users = users[start:start + eval_batch_size]
            seqs, targets = build_eval_sequences(splits, batch_users, max_len=max_len, mode=mode)
            seqs = seqs.to(device)
            targets = targets.to(device)
            scores = model.predict_scores(seqs)
            for row_idx, user in enumerate(batch_users):
                history = splits[user].train if mode == 'val' else (splits[user].train + [splits[user].val])
                target = int(targets[row_idx].item())
                mask_items = [item for item in history if int(item) != target]
                if mask_items:
                    hist_tensor = torch.tensor(mask_items, device=device, dtype=torch.long) - 1
                    scores[row_idx, hist_tensor] = -1e9

            _, top10 = torch.topk(scores, k=10, dim=1)
            top10 = top10 + 1
            top5 = top10[:, :5]
            for row_idx in range(len(batch_users)):
                target = int(targets[row_idx].item())
                pred5 = top5[row_idx].tolist()
                pred10 = top10[row_idx].tolist()
                if target in pred5:
                    hit5 += 1.0
                    ndcg5 += 1.0 / math.log2(pred5.index(target) + 2.0)
                if target in pred10:
                    hit10 += 1.0
                    ndcg10 += 1.0 / math.log2(pred10.index(target) + 2.0)
                total += 1
    if total == 0:
        return {'Recall@5': 0.0, 'Recall@10': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0}
    return {
        'Recall@5': hit5 / total,
        'Recall@10': hit10 / total,
        'NDCG@5': ndcg5 / total,
        'NDCG@10': ndcg10 / total,
    }


def model_stats(model: SASRec) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    domain_dir, data_path = resolve_domain_paths(script_dir, args.domain, args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')
    user_sequences, num_users, num_items = load_sequences(data_path)
    train_stats = count_training_stats(user_sequences, args.max_len, args.train_window_size, args.train_targets)
    splits = build_leave_one_out_splits(user_sequences)
    train_users = [user for user, split in splits.items() if len(split.train) >= 2]
    eval_users = sorted(train_users)

    dataset = SASRecTrainDataset(
        splits=splits,
        max_len=args.max_len,
        train_window_size=args.train_window_size,
        train_targets=args.train_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith('cuda'),
        drop_last=False,
    )

    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_summary = {
        'config': {
            'domain': str(args.domain),
            'domain_dir': str(domain_dir),
            'data_path': str(data_path),
            'device': str(device),
            'model': 'SASRec',
            'max_len': int(args.max_len),
            'hidden_dim': int(args.hidden_dim),
            'num_blocks': int(args.num_blocks),
            'num_heads': int(args.num_heads),
            'dropout': float(args.dropout),
            'batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'epochs': int(args.epochs),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'optimizer': str(args.optimizer),
            'train_targets': str(args.train_targets),
            'train_window_size': int(args.train_window_size),
            'num_negatives': int(args.num_negatives),
            'seed': int(args.seed),
        },
        'data_stats': {'num_users': int(num_users), 'num_items': int(num_items), **train_stats},
        'model_stats': model_stats(model),
        'epoch_logs': [],
        'best_val_metrics': None,
        'best_epoch': None,
        'test_metrics': None,
        'total_train_eval_sec': None,
    }
    print(json.dumps({**run_summary['config'], **run_summary['data_stats']}, ensure_ascii=False, indent=2))

    best_val = -1.0
    best_state = None
    best_epoch = None
    no_improve_evals = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        for step, batch in enumerate(loader, start=1):
            if args.train_targets == 'last_position':
                seq, target = [x.to(device, non_blocking=True) for x in batch]
                hidden = model(seq)[:, -1, :]
                loss = sampled_softmax_loss(hidden, target - 1, model.output_weight(), int(args.num_negatives))
            else:
                seq, pos = [x.to(device, non_blocking=True) for x in batch]
                hidden = model(seq)
                mask = pos.ne(0)
                if mask.sum() == 0:
                    continue
                loss = sampled_softmax_loss(hidden[mask], pos[mask] - 1, model.output_weight(), int(args.num_negatives))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1
            if step % args.log_every == 0:
                print(f'epoch={epoch} step={step} loss={loss.item():.4f}')

        val_metrics = None
        should_stop = False
        if epoch % max(1, int(args.eval_every)) == 0 or epoch == args.epochs:
            val_metrics = evaluate_exact(
                model=model,
                splits=splits,
                users=eval_users,
                max_len=args.max_len,
                eval_batch_size=args.eval_batch_size,
                device=device,
                mode='val',
            )
            val_score = val_metrics[str(args.best_val_metric)]
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                run_summary['best_val_metrics'] = {k: float(v) for k, v in val_metrics.items()}
                no_improve_evals = 0
            else:
                no_improve_evals += 1
                if int(args.early_stop_patience) > 0 and no_improve_evals >= int(args.early_stop_patience):
                    should_stop = True

        epoch_log = {
            'epoch': int(epoch),
            'avg_loss': float(epoch_loss / max(step_count, 1)),
            'val_metrics': val_metrics,
            'epoch_sec': float(time.time() - epoch_start),
            'no_improve_evals': int(no_improve_evals) if val_metrics is not None else None,
            'early_stop_triggered': bool(should_stop),
        }
        run_summary['epoch_logs'].append(epoch_log)
        print(json.dumps(epoch_log, ensure_ascii=False))

        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_exact(
        model=model,
        splits=splits,
        users=eval_users,
        max_len=args.max_len,
        eval_batch_size=args.eval_batch_size,
        device=device,
        mode='test',
    )
    total_sec = time.time() - training_start
    run_summary['best_epoch'] = int(best_epoch) if best_epoch is not None else None
    run_summary['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
    run_summary['total_train_eval_sec'] = float(total_sec)

    print(json.dumps({
        'best_epoch': run_summary['best_epoch'],
        'best_val_metrics': run_summary['best_val_metrics'],
        'test_metrics': run_summary['test_metrics'],
        'total_train_eval_sec': run_summary['total_train_eval_sec'],
    }, ensure_ascii=False, indent=2))

    if str(args.output_json).strip():
        output_dir = domain_dir / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(str(args.output_json)).name
        output_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'saved_json': str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
