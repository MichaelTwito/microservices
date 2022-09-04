import argparse
from datasets import create_dataset
from utils import parse_configuration
import matplotlib.pyplot as plt
import math
from models.neural_network_model import NeuralNetworklModel
import time
import torch
from torch import nn
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
def train(config_file, export=True):
    
    # train_dataset_size = len(train_dataset)
    # print('The number of training samples = {0}'.format(train_dataset_size))

    print('Initializing dataset...')
    # val_dataset = create_dataset(configuration['val_dataset_params'])
    # val_dataset_size = len(val_dataset)
    # print('The number of validation samples = {0}'.format(val_dataset_size))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    # print('Initializing model...')
    # model = create_model(configuration['model_params'])
    # X = torch.rand(1, 28, 28)
    # model = NeuralNetworklModel().to(device)
    # X, Y = make_classification(
    # n_features=4, n_redundant=0, n_informative=3, n_clusters_per_class=2, n_classes=3
    # )
    # plt.title("Multi-class data, 4 informative features, 3 classes", fontsize="large")
    # plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
    # p = nn.Softmax(dim=1)(torch.randn(2,3))
    # print (p)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    # iris = pd.read_csv('./data_dir/Iris.csv')
    
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    data_loader = create_dataset(configuration['train_dataset_params'])
    
    # print(data_loader.dataset.__getitem__(1))
    # print("DEBUG: ")
    # iris['Species'] = iris['Species'].apply(lambda x: mappings[x])
    
    # X = iris.drop(['Species', 'Id'], axis=1).values
    # y = iris['Species'].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # X_train = torch.FloatTensor(X_train)
    # X_test = torch.FloatTensor(X_test)
    # y_train = torch.LongTensor(y_train)
    # y_test = torch.LongTensor(y_test)
    
    model = NeuralNetworklModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    loss_arr = []
    for i in range(epochs):
        for features, label in data_loader.dataset:
            output = model.forward(features)

            preds = torch.max(output)
            print (nn.LogSoftmax(dim=1)(output))
            
            loss = criterion(output[1], label)
            # loss_arr.append(loss)
            if i % 10 == 0:
                print(f'Epoch: {i} Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # torch.Tensor.ndim = property(lambda self: len(self.shape)) 
    loss_arr= [ loss.detach().numpy() for loss in loss_arr]
    epochs_range = range(epochs)
    plt.plot(epochs_range, loss_arr, 'b', label='Train loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.xlabel('Loss')
    plt.legend()
    plt.show()
    preds = []
    test_loss_arr = []
    with torch.no_grad():
        for val in X_test:
            y_hat = model.forward(val)
            test_loss = criterion(y_hat, y_test)
            test_loss_arr.append(test_loss)
            preds.append(y_hat.argmax().item())

    plt.plot(range(X_test), test_loss_arr, 'g', label='Test loss')

    df = pd.DataFrame({'Y': y_test, 'YHat': preds})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    print(df['Correct'].sum() / len(df))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)