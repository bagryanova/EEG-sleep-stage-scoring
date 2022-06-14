import torch
from eegproject.data.utils import TRAIN_SUBJECTS, TEST_SUBJECTS, download_and_preprocess

class UnorderedEEGDataset(torch.utils.data.Dataset):
    def __init__(self, split, preprocessed_path=None, transform=None):
        assert split in ['train', 'test']

        subjects = TRAIN_SUBJECTS if split == 'train' else TEST_SUBJECTS

        self.transform = transform
        if preprocessed_path is None:
            self.X, self.y, self.sequence = download_and_preprocess(subjects)
        else:
            self.X, self.y, self.sequence = torch.load(preprocessed_path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def save(self, file):
        torch.save((self.X, self.y, self.sequence), file)
