import pickle

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import config as hp
from utils import extract_feature, build_LFR_features,save_folder


def pad_collate(batch):
    max_input_len = float('-inf')

    for elem in batch:
        feature, label = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]

    for i, elem in enumerate(batch):
        feature, label = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature

        batch[i] = (padded_input, input_length, label)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class VoxCeleb1Dataset(Dataset):
    def __init__(self, args, split_set):
        self.args = args
        with open(hp.data_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split_set]
        
        self.pre_processed = []
        print('looking in {}'.format(save_folder))
        for sample in data[split_set]:
            split = sample['audiopath'].split('/')
            new_file_path = os.path.join(save_folder,split[-3]+'_'+split[-2]+'_'+split[-1]+'.npy')
            if os.path.exists(new_file_path):
                        self.pre_processed.append({'audiopath':new_file_path,'label':sample['label']})

        print('loading {} {} pre processed samples...'.format(len(self.pre_processed), split_set))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['audiopath']
        label = sample['label']
        feature = extract_feature(input_file=wave, feature='pre_world', dim=hp.n_mels, cmvn=False)
        feature = build_LFR_features(feature, m=hp.LFR_m, n=hp.LFR_n)

        return feature, label

    def __len__(self):
        return len(self.pre_processed)


if __name__ == "__main__":
    import torch
    from utils import parse_args
    from tqdm import tqdm

    args = parse_args()
    train_dataset = VoxCeleb1Dataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers,
                                               collate_fn=pad_collate)

    max_len = 0

    for data in tqdm(train_loader):
        feature = data[0]
        # print(feature.shape)
        if feature.shape[1] > max_len:
            max_len = feature.shape[1]

    print('max_len: ' + str(max_len))
