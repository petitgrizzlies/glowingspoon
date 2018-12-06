from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"features": self.X[idx], "classe": self.y[idx]}
