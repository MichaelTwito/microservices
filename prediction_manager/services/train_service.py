from torch import cuda
def train(model, train_loader,epochs, criterion, optimizer):
    for i in range(epochs):
        model.train()
        for data, label in train_loader:
            if cuda.is_available():
                data, label = data.cuda(), label.cuda()

            label = label.squeeze(1)
            optimizer.zero_grad()
            targets = model(data)
            loss = criterion(targets, label)
            loss.backward()
            optimizer.step()
