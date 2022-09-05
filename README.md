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
    "model_params": {"name": "neural_network_model"},
    "save_model_at": "./saved_models/iris/neural_network_model"
}
```