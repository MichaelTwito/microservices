import torch
import pandas as pd
from .data_service import mappings

def calculate_accuracy(test_loader, model):
    accuracy_tensor = torch.tensor([])
    with torch.no_grad():
        for data, label in test_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            label = label.squeeze(1)
            targets = model(data)
            print(torch.eq(torch.argmax(targets,dim=1), label))
            accuracy_tensor = torch.cat((accuracy_tensor,torch.eq(torch.argmax(targets,dim=1), label)))
    
    return torch.sum(accuracy_tensor)/len(test_loader.dataset)
    

    
    
    
    
    # preds = []
    # with torch.no_grad():
    #    for val in validation_records:
    #        y_hat = model.forward(val)
    #        preds.append(y_hat.argmax().item())
    # return preds

# def calculate_accuracy(validation_labels, preds):
#     df = pd.DataFrame({'Y': validation_labels, 'YHat': preds})
#     df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
#     return df['Correct'].sum() / len(df)
