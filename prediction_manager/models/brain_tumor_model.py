from torch import nn
import torchvision.models as models
class BrainTumorModel(models.efficientnet_b0):
    def __init__(self):
        super().__init__().classifier[1] = nn.Linear(in_features=1280, out_features=10)