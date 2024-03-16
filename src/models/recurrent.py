
from typing import Optional
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    rnn_hidden_states: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class RNN(nn.Module):
    """用于时间序列预测的 RNN 模型"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        self.input_size = input_size # 博文特征
        self.hidden_size = hidden_size # 隐藏维数
        self.predict_size = 3
        
        self.rnn = nn.LSTM(
            input_size=self.input_size + self.predict_size, 
            hidden_size=self.hidden_size
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.predict_size),
        )

        self.loss_fn = nn.MSELoss()

    def forward(
        self, 
        x: Optional[torch.FloatTensor]=None,
        x_len: Optional[torch.LongTensor]=None,
        y: Optional[torch.FloatTensor]=None,
        last_hidden: Optional[torch.FloatTensor]=None
    ) -> ModelOutput:
        """
        两种种前向模式:
        1. 训练, 要求传入 x, x_len, y, 返回 loss 值和隐藏状态 (batch_size, hidden_size)
        2. 回归预测, 要求传入 x 和 last_hidden, 其他将忽略, 返回张量 (batch_size, predict_size)

        Args:
            x (tensor):     shape (seq_len, batch_size, input_size)
            x_len (tensor): shape (batch_size)
            y (tensor):     shape (seq_len, batch_size, predict_size)
            last_hidden (tensor): shape (batch_size, hidden_size)

        Returns:
            ModelOutput: 
        """
        if y is None:
            if x.dim() == 3:
                x = x.squeeze(dim=0)
            logits = self.predictor(torch.concat((x, last_hidden), 1))
            return ModelOutput(
                logits=logits
            )
        else:
            x_len = x_len.cpu()
            rnn_input = torch.concat((x, y), dim=-1)
            packed_seq = pack_padded_sequence(rnn_input, x_len, enforce_sorted=False)
            rnn_out, (h_n, c_n) = self.rnn(packed_seq)
            rnn_out, _ = pad_packed_sequence(rnn_out)
            # rnn_out: (seq_len, batch_size, hidden_size)
            # h_n: (1, batch_size, hidden_size)
            # 这里进 predictor 之前需要将 rnn 输出向右移一位以对齐 y
            rnn_out_previous = rnn_out.roll(shifts=-1, dims=0)
            rnn_out_previous[0].zero_()
            logits = self.predictor(torch.concat((x, rnn_out_previous), -1))

            # 计算损失时，我们需要创建一个掩码来忽略填充元素
            mask = torch.arange(logits.size(0))[:, None] < x_len[None, :]
            mask = mask.to(logits.device)  # 确保掩码在正确的设备上
            loss = self.loss_fn(logits * mask.unsqueeze(-1), y * mask.unsqueeze(-1))

            return ModelOutput(
                loss=loss,
                last_hidden_state=h_n,
                rnn_hidden_states=rnn_out,
                logits=logits
            )
