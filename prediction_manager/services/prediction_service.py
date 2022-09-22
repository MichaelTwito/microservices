import base64
from io import BytesIO
from  PIL.Image import open
from torch import Tensor, no_grad, unsqueeze
from datasets.iris_dataset import iris_mappings
from datasets.brain_tumor_dataset import bt_mappings
from datasets.brain_tumor_dataset import get_validation_transform

def predict_species_by_properties(model, properties):
    sample_tensor=Tensor(\
    [properties['SepalLengthCm'],properties['SepalWidthCm'],\
     properties['PetalLengthCm'],properties['PetalWidthCm']])
    
    with no_grad():
        predicted_species_index = model.forward(sample_tensor).argmax().item()
        species = list(iris_mappings().keys())\
                    [list(iris_mappings().values()).index(predicted_species_index)]
        return species


def predict_type_of_img(model, properties):
    decoded_img = base64.b64decode(properties['Base64Image'])
    file_like = BytesIO(decoded_img)
    tensor = get_validation_transform(224)(open(file_like))
    tensor = unsqueeze(tensor, dim=0)
    with no_grad(): 
        predicted_type_index = model.forward(tensor).argmax(dim=1)
        type = list(bt_mappings().keys())\
                    [list(bt_mappings().values()).index(predicted_type_index)]
    return type