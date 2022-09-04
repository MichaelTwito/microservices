import torch
from datetime import datetime
from services.train_service import train
from services.data_service import prepare_data
from services.validate_service import calculate_accuracy
from services.model_service import create_model, get_criterion, get_optimizer, load_model as load_model_service
from services.prediction_service import predict_species_by_properties
from datasets.iris_dataset import TrainIrisDataset
from torchvision import datasets, transforms
# from datasets.data_loader import MyCollate


def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=False):
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,pin_memory=pin_memory)
    return loader


def train_model(dataset_path, epochs, optimizer_params, criterion, model_params, save_trained_model_at):
    Model = create_model(model_params['name'])
    Criterion = get_criterion(criterion)
    Optimizer = get_optimizer(optimizer_params['algorithm'])

    # train_records, validation_records,\
    #     train_labels, validation_labels\
    #         = prepare_data(dataset_path)

    train_dataset = TrainIrisDataset(dataset_path)
    
    train_data, test_data = random_split(train_dataset,[120,30])

    train_loader = get_train_loader(train_data, 30)
    test_loader = get_train_loader(test_data, 30)

    train_loader, test_loader = prepare_data(dataset_path)
    train(Model,train_loader, test_loader,\
        epochs = epochs,\
        criterion = Criterion(),\
        optimizer = Optimizer(Model.parameters(), float(optimizer_params['lr'])))

    saved_model_path = ''

    if save_trained_model_at != '':
        dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        saved_model_path = save_trained_model_at + '_' + dt_string + '.pth'
        torch.save(Model.state_dict(), saved_model_path)
    
    return calculate_accuracy(test_loader, Model), saved_model_path

def load_model(*args):
    return load_model_service(*args)

def predict(*args):
    return predict_species_by_properties(*args)