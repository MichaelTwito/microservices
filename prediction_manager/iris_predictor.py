
# import torch
# # from torch.optim import Adam
# # import torch.nn.CrossEntropyLoss
# # from torch.nn import CrossEntropyLoss
# # import torch.nn as nn
# # import torch.nn.functional as F
# import pandas as pd
# # import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from models.neural_network_model import NeuralNetworkModel
# import sys
# import importlib

# def str_to_class(classname):
#     return getattr(sys.modules[__name__], classname)

# def get_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"

# def mappings():
#     return {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}

# def prepare_data(path):
#     iris = pd.read_csv(path)
#     iris['Species'] = iris['Species'].apply(lambda x: mappings()[x])
    
#     X = iris.drop(['Species', 'Id'], axis=1).values
#     y = iris['Species'].values

#     train_records, validation_records, train_labels, validation_labels = train_test_split(X, y, test_size = 0.2, random_state = 42)
#     train_records = torch.FloatTensor(train_records)
#     validation_records = torch.FloatTensor(validation_records)
#     train_labels = torch.LongTensor(train_labels)
#     validation_labels = torch.LongTensor(validation_labels)
#     return train_records, validation_records, train_labels, validation_labels

# def train(model, train_records,train_labels,epochs, criterion, optimizer):
#     loss_arr = []

#     for i in range(epochs):
#        y_hat = model.forward(train_records)
#        loss = criterion(y_hat, train_labels)
#        loss_arr.append(loss)
    
#        if i % 10 == 0:
#         #    print(f'Epoch: {i} Loss: {loss}')
#         print("Epoch: {0}, loss: {1}".format(i, loss))
    
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()

# def get_validation_predictions(validation_records, model):
#     preds = []
#     with torch.no_grad():
#        for val in validation_records:
#            y_hat = model.forward(val)
#            preds.append(y_hat.argmax().item())
#     return preds

# def calculate_accuracy(validation_labels, preds):
#     df = pd.DataFrame({'Y': validation_labels, 'YHat': preds})
#     df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
#     return df['Correct'].sum() / len(df)

# def predict_species_by_properties(properties, model):
#     sample_tensor=torch.Tensor(\
#     [properties['SepalLengthCm'],properties['SepalWidthCm'],\
#      properties['PetalLengthCm'],properties['PetalWidthCm']])
    
#     with torch.no_grad():
#         predicted_species_index = model.forward(sample_tensor).argmax().item()
#         species = list(mappings().keys())\
#                     [list(mappings().values()).index(predicted_species_index)]
#         return species

# def build_model_class_name(model_name):
#     return ''.join(map(str,list\
#            (map(lambda x: x.capitalize(),\
#            model_name.split('_')))))

# def get_dynamically_imported_class(from_module, class_name):
#     module = importlib.import_module(from_module)
#     return getattr(module, class_name) 

# def create_classes_from_strings(model_name, criterion, optimizer_params):
#     model_class_name = build_model_class_name(model_name)
#     return [get_dynamically_imported_class(x,y) for x,y in 
#                 zip([('models.' + model_name), 'torch.nn','torch.optim'],\
#                     [model_class_name, criterion, optimizer_params['algorithm']])]
#     device = get_device()

#     train_records, validation_records,\
#         train_labels, validation_labels\
#             = prepare_data(dataset_path)
    
#     ModelClass, Criterion, Optimizer = \
#         create_classes_from_strings(model_params['name'], criterion, optimizer_params)

#     model = ModelClass().to(device)
    
#     train(model,train_records, train_labels,\
#         epochs = epochs,\
#         criterion = Criterion(),\
#         optimizer = Optimizer(model.parameters(), float(optimizer_params['lr'])))
    
#     if save_model_at:    
#         torch.save(model.state_dict(), save_model_at + '.pth')

#     preds = get_validation_predictions(validation_records, model)
#     accuracy = calculate_accuracy(validation_labels, preds)

#     return accuracy

# def load_model(class_name, path_to_model):
#     model = get_dynamically_imported_class('torch.nn', class_name)
#     model.load_state_dict(torch.load(path_to_model))
#     return model

# if __name__ == '__main__':
#     create_and_train_model('./data_dir/Iris.csv', 100, {"algorithm": "Adam", "lr": 0.01}, "CrossEntropyLoss",{"name": "neural_network_model"}, './models/iris_prediction_model')
#     # create_network('./data_dir/Iris.csv', 100, {"type": "NeuralNetworkModel"})