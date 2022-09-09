import torch
import pandas as pd
from torch.utils.data import Dataset

def iris_mappings(): 
    return {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}

class IrisDataset(Dataset):
    """Represents an iris dataset.
    """

    def __init__(self, csv_file):
        csv_df = pd.read_csv(csv_file)
        self.length = len(csv_df.index)
        self.features = torch.Tensor(csv_df.iloc[:, 1:-1].values)
        self.labels = torch.Tensor(csv_df.iloc[:,-1].apply(lambda x: iris_mappings()[x])).long()
        

    def __getitem__(self, index):
        item =  self.features[index]
        label = self.labels[index]
        return item, label

    def __len__(self):
        return self.length
        