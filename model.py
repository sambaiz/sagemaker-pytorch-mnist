import torch
from torch import nn, cuda
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, dropout):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)  # -> 24x24
        self.pool1 = nn.MaxPool2d(2)  # -> 12x12
        self.conv2 = nn.Conv2d(64, 128, 5)  # -> 8x8
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        return F.relu(self.dense(x))


def _average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def train(model, train_loader, device, is_distributed, lr, momentum):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if is_distributed and not cuda.is_available():
            # average gradients manually for multi-machine cpu case only
            _average_gradients(model)
        optimizer.step()


def test(model, test_loader, device):
    model.eval()
    correct = 0
    loss_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += F.nll_loss(output, target, reduction='sum').cpu().data  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    loss_avg = loss_sum / len(test_loader.dataset)
    return accuracy, loss_avg
