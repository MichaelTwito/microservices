import torch

def calculate_accuracy(test_loader, model):
    accuracy_tensor = torch.tensor([])
    with torch.no_grad():
        for data, label in test_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            targets = model(data)
            accuracy_tensor = torch.cat((accuracy_tensor,torch.eq(torch.argmax(targets,dim=1), label)))
    
    return torch.sum(accuracy_tensor)/len(test_loader.dataset)
