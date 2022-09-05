from torch import Tensor, no_grad
from datasets.iris_dataset import iris_mappings

def predict_species_by_properties(model, properties):
    sample_tensor=Tensor(\
    [properties['SepalLengthCm'],properties['SepalWidthCm'],\
     properties['PetalLengthCm'],properties['PetalWidthCm']])
    
    with no_grad():
        predicted_species_index = model.forward(sample_tensor).argmax().item()
        species = list(iris_mappings().keys())\
                    [list(iris_mappings().values()).index(predicted_species_index)]
        return species


