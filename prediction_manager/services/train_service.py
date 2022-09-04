from datasets.iris_dataset import TrainIrisDataset
from torch.utils.data import DataLoader
import torch
import numpy as num
def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=False):
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory)
    return loader


def train(model, train_loader,test_loader,epochs, criterion, optimizer):
    for i in range(epochs):
        model.train()     
        for data, label in train_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()

            label = label.squeeze(1)
            optimizer.zero_grad()
            targets = model(data)
            loss = criterion(targets, label)
            loss.backward()
            optimizer.step()
            # trainloss += loss.item()
            # print(trainloss)
        # testloss = 0.0
        # model.eval()    
        # for data, label in test_loader:
        #     if torch.cuda.is_available():
        #         data, label = data.cuda(), label.cuda()

        #     label = label.squeeze(1)
        #     targets = model(data)
        #     loss = criterion(targets,label)
        #     testloss = loss.item() * data.size(0)

        # print(f'Epoch {i+1} \t\t Training data: {trainloss / len(train_loader)} \t\t Test data: {testloss / len(test_loader)}')
        # if minvalid_loss > testloss:
        #     print(f'Test data Decreased({minvalid_loss:.6f}--->{testloss:.6f}) \t Saving The Model')
        #     minvalid_loss = testloss

    # loss_arr = []

    # for i in range(epochs):
    #    y_hat = model.forward(train_records)
    #    loss = criterion(y_hat, train_labels)
    #    loss_arr.append(loss)
    
    #    if i % 10 == 0:
    #     #    print(f'Epoch: {i} Loss: {loss}')
    #     print("Epoch: {0}, loss: {1}".format(i, loss))
    
    #    optimizer.zero_grad()
    #    loss.backward()
    #    optimizer.step()

# def train_one_epoch(epoch_index, tb_writer):
#     train_dataset = TrainIrisDataset('./data_dir/Iris.csv')
#     training_loader = get_train_loader(train_dataset, 32)
    
#     loss_fn = torch.nn.CrossEntropyLoss()
#     for i, data in enumerate(training_loader):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.
