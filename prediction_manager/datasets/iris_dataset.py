import torch
import pandas as pd
from torch.utils.data import Dataset

def iris_mappings():
    return {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}

class IrisDataset(Dataset):
    """Represents an iris dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, csv_file,transform=None):
        self.csv_df = pd.read_csv(csv_file)
        self.label_mappings = iris_mappings()

    def __getitem__(self, index):
        item =  torch.Tensor(self.csv_df.iloc[index, 1:-1])
        label = torch.Tensor(self.csv_df.iloc[index,-1:].apply(lambda x: self.label_mappings[x])).long()
        return item, label

    def __len__(self):
        return len(self.csv_df.index)
        