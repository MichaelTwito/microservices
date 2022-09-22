import torch
from .helper import get_dynamically_imported_class, build_class_name

def create_model(model_name):
    model_class_name = build_class_name(model_name)
    ModelClass = get_dynamically_imported_class(\
             ('models.' + model_name), model_class_name, 'prediction_manager')     
    return ModelClass(pretrained=True).to(get_device()) if model_class_name == "BrainTumorModel" else ModelClass().to(get_device())
    

def get_criterion(criterion):
    return get_dynamically_imported_class('torch.nn', criterion)

def get_optimizer(optimizer):
    return get_dynamically_imported_class('torch.optim', optimizer)
    
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path_to_model):
    model_name = path_to_model[:-24].rsplit('/', 1)[-1]
    ClassName = build_class_name(model_name)
    model = get_dynamically_imported_class(('models.' + model_name), ClassName, 'prediction_manager')()
    model.load_state_dict(torch.load(path_to_model))
    return model

def create_classes_from_strings(model_name, criterion, optimizer_params):
    model_class_name = build_class_name(model_name)
    return [get_dynamically_imported_class(x,y) for x,y in 
                zip([('models.' + model_name), 'torch.nn','torch.optim'],\
                    [model_class_name, criterion, optimizer_params['algorithm']])]

