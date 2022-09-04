import torch
from .data_service import mappings
def predict_species_by_properties(model, properties):
    sample_tensor=torch.Tensor(\
    [properties['SepalLengthCm'],properties['SepalWidthCm'],\
     properties['PetalLengthCm'],properties['PetalWidthCm']])
    
    with torch.no_grad():
        predicted_species_index = model.forward(sample_tensor).argmax().item()
        species = list(mappings().keys())\
                    [list(mappings().values()).index(predicted_species_index)]
        return species


