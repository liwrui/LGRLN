import torch
from torch.utils.data import Dataset
import json
import numpy as np


class VideoXumDataset(Dataset):
    def __init__(self, mode='train', fea='blip', tau=2, device=torch.device('cuda')):
        """
        :param mode:   train, test, val
        :param fea:    blip, vt_clipscore
        """
        self.menu_pth = r"../videoxum/{}_videoxum.json".format(mode)
        self.feature_pth = r"../videoxum/{}".format(fea)
        self.fea = fea
        with open(self.menu_pth, 'r', encoding='utf-8') as f:
            self.menu = json.load(f)
        self.tau = tau
        self.device = device

    def __getitem__(self, item):
        line = self.menu[item]
        vid = line['video_id']
        vsum_onehot = line['vsum_onehot']  # 10 labels

        feature = np.load(f"{self.feature_pth}/{vid}.npz")
        if self.fea == 'blip':
            feature = feature['features']
        else:
            feature = feature['vision']

        edges = [[], []]
        edge_attr = []

        for i in range(feature.shape[0]):
            for j in range(max([0, i-self.tau]), min([feature.shape[0]-1, i + self.tau])):
                edges[0].append(i)
                edges[1].append(j)
                if i > j:
                    edge_attr.append(1)
                elif i < j:
                    edge_attr.append(-1)
                else:
                    edge_attr.append(0)

        return torch.tensor(feature, dtype=torch.float, device=self.device), torch.tensor(vsum_onehot, dtype=torch.float, device=self.device), \
               torch.tensor(edges, dtype=torch.long, device=self.device), torch.tensor(edge_attr, device=self.device)

    def __len__(self):
        return len(self.menu)

        # print(blip['features'].shape, vt_clip['vision'].shape, vt_clip['text'].shape, sampled_frames)


if __name__ == '__main__':
    for data in VideoXumDataset():
        x, y, e, e_attr = data
        print(x.shape, y.shape, e.shape, e_attr.shape)

