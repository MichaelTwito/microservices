### Create,Train and Validate request
 ```
 POST http://localhost:5000/api/model/train
 ```
```json
{
    "dataset_params": {"path": "./data_dir/Iris.csv", "train_test_split_ratio": "0.2"},
    "train_params": {"epochs": "100","batch_size": "30","num_of_workers": "0"},
    "test_params": {"batch_size": "30","num_of_workers": "0"},
    "optimizer_params": {"algorithm": "Adam", "lr": "0.01"},
    "criterion": "CrossEntropyLoss", 
    "model_params": {"name": "neural_network_model", "pretrained": "False"},
    "save_model_at": "./saved_models/iris/neural_network_model"
}
```

```json
{
    "dataset_params": {"name": "brain_tumor_dataset","path": "./data_dir/brain_tumor", "train_test_split_ratio": "0.2"},
    "train_params": {"epochs": "35","batch_size": "34","num_of_workers": "8"},
    "test_params": {"batch_size": "34","num_of_workers": "8"},
    "optimizer_params": {"algorithm": "Adam", "lr": "0.01"},
    "criterion": "CrossEntropyLoss", 
    "model_params": {"name": "brain_tumor_model", "pretrained": "True"},
    "save_model_at": "./saved_models/brain_tumor/brain_tumor_model"
}
```
### Run saved model for prediction request
```
POST http://localhost:5000/api/predictions/nerual_networks/predict_iris_species
```
```json
{
"SepalLengthCm": 5.4,
"SepalWidthCm": 0.2,
"PetalLengthCm": 3.4,
"PetalWidthCm": 0.5,
"PathToModel": "./saved_models/iris/neural_network_model_05_09_2022_14_58_24.pth"
}
```