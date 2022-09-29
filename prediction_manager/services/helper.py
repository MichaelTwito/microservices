import importlib
def get_dynamically_imported_class(from_module, class_name, package=None):
    module = importlib.import_module(from_module, package=package)
    return getattr(module, class_name) 


def build_class_name(model_name):
    return ''.join(map(str,list\
           (map(lambda x: x.capitalize(),\
           model_name.split('_')))))

def string_to_bool(string):
    return string == 'True'