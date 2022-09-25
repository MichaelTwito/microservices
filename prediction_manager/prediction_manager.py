from torch import save
from datetime import datetime
from services.train_service import train
from services.data_service import prepare_data
from services.validate_service import calculate_accuracy
from services.prediction_service import predict_species_by_properties, predict_type_of_img
from services.model_service import create_model, get_criterion, get_optimizer, load_model as load_model_service

def train_model(dataset_params, train_params, test_params, optimizer_params, criterion, model_params, save_trained_model_at):
    Model = create_model(model_params)
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

def predict(model, request_dict, prediction_type):
    if prediction_type == "iris":
        return predict_species_by_properties(model, request_dict)
    else:
        return predict_type_of_img(model, request_dict)
    
