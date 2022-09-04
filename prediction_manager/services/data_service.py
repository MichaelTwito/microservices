import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.iris_dataset import TrainIrisDataset
# from datasets.data_loader import MyCollate
from torch.utils.data import DataLoader, random_split

def mappings():
    return {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}


    # If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=False):
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory)
    return loader


def prepare_data(path):
    iris = pd.read_csv(path)
    dataset = TrainIrisDataset(path)
    train_data, test_data = random_split(dataset,[120,30])
    train_loader = get_train_loader(train_data, 30)
    test_loader = get_train_loader(test_data, 30)
    # print(train[10])
    # iris['Species'] = iris['Species'].apply(lambda x: mappings()[x])
    # 
    # X = iris.drop(['Species', 'Id'], axis=1).values
    # y = iris['Species'].values
# 
    # train_records, validation_records, train_labels, validation_labels = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # train_records = torch.FloatTensor(train_records)
    # validation_records = torch.FloatTensor(validation_records)
    # train_labels = torch.LongTensor(train_labels)
    # validation_labels = torch.LongTensor(validation_labels)
    # return train_records, validation_records, train_labels, validation_labels



