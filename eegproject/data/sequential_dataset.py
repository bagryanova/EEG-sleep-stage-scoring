import torch
from eegproject.data.utils import TRAIN_SUBJECTS, TEST_SUBJECTS, download_and_preprocess
from eegproject.data.transforms import scale
import math
from torch.utils.data import DataLoader

class SequentialEEGDataset(torch.utils.data.Dataset):
    def __init__(self, split, preprocessed_path=None, transform=None):
        assert split in ['train', 'test']

        subjects = TRAIN_SUBJECTS if split == 'train' else TEST_SUBJECTS

        self.transform = transform
        if preprocessed_path is None:
            X, y, sequence = download_and_preprocess(subjects)
        else:
            X, y, sequence = torch.load(preprocessed_path)

        self.X = []
        self.y = []
        self.idx = []
        for i in torch.unique(sequence):
            self.X.append(scale(X[sequence == i]))
            self.y.append(y[sequence == i])
            self.idx.append(i.item())

        self.idx = torch.tensor(self.idx)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.transform(self.X[idx]), self.y[idx]

        X, y = [], []
        for i in idx:
            if self.transform is not None:
                X.append(self.transform(self.X[i]))
                y.append(self.y[i])

        return X, y

def get_short_sequence_dataset(dataset, sequence_length, max_offset=0, skip=0.0):
    res_X = []
    res_y = []

    for X, y in zip(dataset.X, dataset.y):
        for start_ind in range(torch.randint(0, max_offset + 1, size=(1,)), X.shape[0], sequence_length):
            end_ind = min(X.shape[0], start_ind + sequence_length)

            l = end_ind - start_ind
            idx = torch.randperm(l)[:(l - math.ceil(l * skip))]

            res_X.append(dataset.transform(X[start_ind:end_ind][idx]))
            res_y.append(y[start_ind:end_ind][idx])

    res_X = torch.nn.utils.rnn.pad_sequence(res_X, batch_first=True, padding_value=-1)
    res_y = torch.nn.utils.rnn.pad_sequence(res_y, batch_first=True, padding_value=-1)
    return res_X, res_y

def iterate_batches(dataset, batch_size, sequence_length, max_offset, skip=0.0):
    X, y = get_short_sequence_dataset(dataset, sequence_length, max_offset=max_offset, skip=skip)

    dataset_ = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=True)

    for x in dataloader:
        yield x
