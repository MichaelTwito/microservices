from torch import nn
from torchvision.models import efficientnet_b0
class BrainTumorModel(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.model = efficientnet_b0(pretrained)
    
    def forward(self, data):
        return self.model.forward(data)

