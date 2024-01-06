from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, input_tensors, target_tensors):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.target_tensors[idx]