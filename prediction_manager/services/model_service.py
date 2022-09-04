import torch
import importlib


def create_model(model_name):
    device = get_device()
    model_class_name = build_model_class_name(model_name)
    ModelClass = get_dynamically_imported_class(('models.' + model_name), model_class_name)
    return ModelClass().to(device)
    
def get_criterion(criterion):
    return get_dynamically_imported_class('torch.nn', criterion)

def get_optimizer(optimizer):
    return get_dynamically_imported_class('torch.optim', optimizer)
    
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path_to_model):
    model_name = path_to_model[:-24].rsplit('/', 1)[-1]
    ClassName = build_model_class_name(model_name)
    model = get_dynamically_imported_class(('models.'+model_name), ClassName, 'prediction_manager')()
    model.load_state_dict(torch.load(path_to_model))
    return model

def get_dynamically_imported_class(from_module, class_name, package=None):
    module = importlib.import_module(from_module, package='prediction_manager')
    return getattr(module, class_name) 

def create_classes_from_strings(model_name, criterion, optimizer_params):
    model_class_name = build_model_class_name(model_name)
    return [get_dynamically_imported_class(x,y) for x,y in 
                zip([('models.' + model_name), 'torch.nn','torch.optim'],\
                    [model_class_name, criterion, optimizer_params['algorithm']])]

def build_model_class_name(model_name):
    return ''.join(map(str,list\
           (map(lambda x: x.capitalize(),\
           model_name.split('_')))))

