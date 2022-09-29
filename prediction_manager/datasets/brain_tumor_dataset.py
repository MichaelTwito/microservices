import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def bt_mappings(): 
    return {'glioma': 0,'meningioma': 1,'no_tumor': 2, 'pituitary': 3}


def get_train_transform(image_size):
    return transforms.Compose([
           transforms.Resize((image_size, image_size)),
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomVerticalFlip(p=0.5),
           transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5)),
           transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
           transforms.ToTensor(),
           transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229, 0.224, 0.225]
           )])

def get_validation_transform(IMAGE_SIZE):
    return transforms.Compose([
           transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
           transforms.ToTensor(),
           transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )])

class BrainTumorDataset(Dataset):
    """Represents a Brain Tumor dataset.
    """

    def __init__(self, dir):
        self.dataset = datasets.ImageFolder(dir, transform=get_train_transform(224))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self): 
        return len(self.dataset)
        