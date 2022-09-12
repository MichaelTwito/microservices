from torch import save
from datetime import datetime
from services.train_service import train
from services.data_service import prepare_data
from services.validate_service import calculate_accuracy
from services.prediction_service import predict_species_by_properties
from services.model_service import create_model, get_criterion, get_optimizer, load_model as load_model_service
from datasets.brain_tumor_dataset import BrainTumorDataset

def train_model(dataset_params, train_params, test_params, optimizer_params, criterion, model_params, save_trained_model_at):
    Model = create_model(model_params['name'])
    Criterion = get_criterion(criterion)
    Optimizer = get_optimizer(optimizer_params['algorithm'])

    train_loader, test_loader = prepare_data(dataset_params, train_params, test_params)

    train(Model,train_loader,\
        epochs = int(train_params['epochs']),\
        criterion = Criterion(),\
        optimizer = Optimizer(Model.parameters(), float(optimizer_params['lr'])))

    saved_model_path = save_model(Model,save_trained_model_at)

    return calculate_accuracy(test_loader, Model), saved_model_path

def save_model(Model,save_trained_model_at):
    saved_model_path = ''
    if save_trained_model_at != '':
        dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        saved_model_path = save_trained_model_at + '_' + dt_string + '.pth'
        save(Model.state_dict(), saved_model_path)
    return saved_model_path

def load_model(*args):
    return load_model_service(*args)

def predict(*args):
    return predict_species_by_properties(*args)

def run(model_params, train_params, test_params, dataset_params, criterion, optimizer_params): 
    Model = create_model(model_params['name'])
    Criterion = get_criterion(criterion)
    Optimizer = get_optimizer(optimizer_params['algorithm'])

    train_loader, test_loader = prepare_data(dataset_params, train_params, test_params)
    train(Model,train_loader,\
        epochs = int(train_params['epochs']),\
        criterion = Criterion(),\
        optimizer = Optimizer(Model.parameters(), float(optimizer_params['lr'])))

    return 

if __name__ == "__main__":
    dataset_params = {'name': 'brain_tumor_dataset','path': './data_dir/brain_tumor', 'train_test_split_ratio': '0.2'}
    train_params = {'epochs': '5','batch_size': '34','num_of_workers': '8'}
    test_params = {'batch_size': '30','num_of_workers': '8'}
    model_params = {'name': 'efficientnet_b0',}
    optimizer_params = {'algorithm': 'Adam', 'lr': '0.01'}
    criterion = "CrossEntropyLoss"
    save_trained_model_at =  "./saved_models/brain_tumor/brain_tumor_model"
    result = train_model(dataset_params, train_params, test_params, optimizer_params, criterion, model_params, save_trained_model_at)
    print(result)