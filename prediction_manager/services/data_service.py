from torch.utils.data import DataLoader, random_split
from .helper import get_dynamically_imported_class, build_class_name
def get_loader(dataset, params):
    batch_size, num_workers = map(int, [params['batch_size'], params['num_of_workers']])
    return DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=True, pin_memory=False)
    
def calculate_size(dataset_size: int, train_test_split_ratio: float):
    test_dataset_size = int(dataset_size*train_test_split_ratio)
    train_dataset_size = int(dataset_size - test_dataset_size)
    return train_dataset_size, test_dataset_size

def prepare_data(dataset_params: dict, train_params: dict, test_params:dict):
    dataset_class_name = build_class_name(dataset_params['name'])  
    DatasetClass = get_dynamically_imported_class(\
                   from_module=('datasets.' + dataset_params['name']),\
                   class_name=dataset_class_name,
                   package='prediction_manager')\
                   (dataset_params['path'])

    train_dataset_size,\
         test_dataset_size = calculate_size(\
                                len(DatasetClass), float(dataset_params['train_test_split_ratio']))
    train_data,\
         test_data = random_split(DatasetClass,[train_dataset_size,test_dataset_size])
    print(len(train_data))
    print(len(test_data))
    return get_loader(train_data, train_params),\
           get_loader(test_data,test_params)

