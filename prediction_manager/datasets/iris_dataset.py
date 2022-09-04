# from cv2 import imread
# # from datasets.base_dataset import get_transform
# from datasets.base_dataset import BaseDataset
from torchvision import datasets
import os
import pandas as pd
# from skimage import io, transform
import torch
from torch.utils.data import Dataset

class TrainIrisDataset(Dataset):
    """Represents an iris dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, csv_file,transform=None):
        self.csv_df = pd.read_csv(csv_file)
        self.label_mappings = {
                             'Iris-setosa': 0,
                             'Iris-versicolor': 1,
                             'Iris-virginica': 2
                              }
        # self.image_folder = datasets.ImageFolder(os.path.join(configuration['dataset_path'], configuration['mode']), transform=transform)

    def __getitem__(self, index):
        item =  torch.Tensor(self.csv_df.iloc[index, 1:-1])
        label = torch.Tensor(self.csv_df.iloc[index,-1:].apply(lambda x: self.label_mappings[x])).long()
        return item, label

    def __len__(self):
        return len(self.csv_df.index)
        