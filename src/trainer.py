"""Standalone trainer for training a torch model."""

import time
from dataclasses import dataclass
import pandas
from tqdm import tqdm
from loguru import logger

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_


@dataclass
class TrainingArguments:
    device: str = 'cuda:0'
    epochs: int = 32
    batch_size: int = 64
    learning_rate: float = 0.01
    clip_grad_norm: float = 5.0
    lr_gamma: float = 0.95


def collate_fn(batch):
    """
    batch: 一个由元组组成的列表，每个元组对应一个数据点，包含了x_tensor, y_tensor 和 x_len。
    """
    # 解包每个样本中的数据
    x_tensors = [x for x, _, _ in batch]
    y_tensors = [y for _, y, _ in batch]
    x_lens = [x_len for _, _, x_len in batch]

    # 对x进行padding，使得每个序列都有相同的长度
    padded_x_tensors = pad_sequence(x_tensors, batch_first=False, padding_value=0.0)
    padded_y_tensors = pad_sequence(y_tensors, batch_first=False, padding_value=0.0)
    x_lens = torch.stack(x_lens)
    x_lens = x_lens.squeeze(1)
    return padded_x_tensors, x_lens, padded_y_tensors


class Trainer:
    def __init__(self, trainset, validset, testset, model, args: TrainingArguments):
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_gamma)
        self.args = args

        self.model.to(self.args.device)

    def train(self):
        logger.debug(self.model)

        train_dataloader = DataLoader(self.trainset, self.args.batch_size, shuffle=True, collate_fn=collate_fn)

        best_dev_f1, best_dev_loss = None, None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            progress_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}', disable=False)
            # epoch_start_time = time.perf_counter()
            self.model.train()
            for x, x_len, y in train_dataloader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                self.model.zero_grad()
                outputs = self.model(x, x_len, y)
                outputs.loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_norm)
                self.optimizer.step()
                progress_bar.set_postfix({
                    'loss': round(outputs.loss.item(), 4),
                    'lr': round(self.scheduler.get_last_lr()[0], 6)
                })
                progress_bar.update(1)
            self.scheduler.step()
            progress_bar.close()

            # dev_f1, dev_loss = self.evaluate()
            # epoch_time_cost = time.perf_counter() - epoch_start_time
        #     logger.info(f'Epoch {epoch}, training_time: {epoch_time_cost:.2f}')
        #     if best_dev_f1 is None or dev_f1 > best_dev_f1:
        #         best_dev_f1 = dev_f1
        #         best_dev_loss = dev_loss
        #         torch.save(
        #             self.model.state_dict(),
        #             f'{self.args.save_dir}/best_model.bin'
        #         )
        #         logger.info(f"Best F1 Reached ({best_dev_f1}). Model is saved to model_checkpoints/best_model.bin")
            
        # logger.info(f'Validation Set: best F1 = {best_dev_f1:.4f}, best loss = {best_dev_loss:.6f}')
        return None
