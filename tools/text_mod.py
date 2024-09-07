import argparse
import glob
import os
import torch
from gravit.utils.parser import get_cfg
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import model as M


# tokenizer = AutoTokenizer.from_pretrained(r"../bert")
#
# model = AutoModel.from_pretrained(r"../bert")
#
# ids = tokenizer.encode("who are you?", return_tensors="pt")
#
#
# out = model(ids)
#
# print(out.last_hidden_state.shape)


class TextMod(nn.Module):
    def __init__(self, v_size, t_size=768, n_head=8, dropout=0.1):
        super(TextMod, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(r"./bert")
        self.bert = AutoModel.from_pretrained(r"./bert")
        self.q = nn.Linear(v_size, v_size // n_head * n_head)
        self.k = nn.Linear(t_size, v_size // n_head * n_head)
        self.v = nn.Linear(t_size, v_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(v_size),
            nn.Linear(v_size, v_size),
            nn.GELU(),
            nn.Linear(v_size, v_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head

    def forward(self, video, text):
        text = self.bert(self.tokenizer.encode(text, return_tensors="pt").to(video.device)).last_hidden_state[0]    # s2, h

        s2, _ = text.shape
        s1, _ = video.shape

        q = self.q(video).reshape(s1, self.n_head, -1).transpose(0, 1)       # n, s1, k
        k = self.k(text).reshape(s2, self.n_head, -1).transpose(0, 1)        # n, s2, k

        attn = (q @ k.transpose(1, 2)) * (1 / _ ** 0.5)   # n, s1, s2
        attn = torch.softmax(attn, dim=2)    # n, s1, s2

        attn = self.dropout(attn)

        v = self.v(text)    # s2, h
        v = attn @ v   # n, s1, h
        v = torch.mean(v, dim=0)   # s1, h
        o = v * 0.5 + video * 0.5
        o = self.dropout(o)
        o = self.ffn(o)

        return  o * 0.5 + video * 0.5


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str, help='Root directory to the data', default='./data')
    parser.add_argument('--root_result', type=str, help='Root directory to output', default='./results')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment', required=True)
    parser.add_argument('--eval_type', type=str, help='Type of the evaluation', required=True)
    parser.add_argument('--split', type=int, help='Split to evaluate')
    parser.add_argument('--all_splits', action='store_true', help='Evaluate all splits')

    device = torch.device("cuda")

    args = parser.parse_args()

    path_result = os.path.join(args.root_result, args.exp_name)

    results = []
    if args.all_splits:
        results = glob.glob(os.path.join(path_result, "*", "cfg.yaml"))
    else:
        if not os.path.isdir(path_result):
            raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

        results.append(os.path.join(path_result, 'cfg.yaml'))

    for result in results:
        args.cfg = result
        cfg = get_cfg(args)

        model = M.SPELL(cfg, cfg['t_emb']).to(device)
        mod = TextMod()



